"""Inférence sans réentraînement : dossier d'images PNG -> CSV Id,Label.

    uv run python src/predict.py --images test --stack deploy_B --out predictions.csv

Rejoue toute la chaîne à partir des poids sauvegardés dans outputs/models :
  1. métadonnées PNG et statistiques d'image ;
  2. embeddings gelés (DINOv2 / CLIP) et features kNN contre le train ;
  3. CNN fine-tunés (5 folds chacun, TTA flip) ;
  4. stacking LightGBM (5 folds) ;
  5. copie du label pour les doublons exacts d'une image du train.

Stacks disponibles (recette dans outputs/models/lgb_<stack>_config.json) :
  deploy_B : ConvNeXt 160 + ConvNeXt 224/160 pseudo-labellisés + kNN + métadonnées (OOF 98,42 %)
  deploy_A : ConvNeXt 160 + kNN + métadonnées, sans pseudo-labels (OOF 98,05 %)

Nécessite outputs/index/index.parquet et outputs/embeddings/*.npy (référence kNN = train).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import timm
import torch
from tqdm import tqdm

import config as C
from concepts import _knn_block
from dataset import _load_resized, to_tensor
from embed import embed
from meta_features import extract
from utils import load_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cnn_probs(name, paths, bs=64):
    """Moyenne des 5 folds d'un CNN, avec TTA flip horizontal."""
    arch, res = C.CNN_MODELS[name]
    weights = [C.MODELS_DIR / f"{name}_f{k}.pt" for k in range(C.N_FOLDS)]
    missing = [w.name for w in weights if not w.exists()]
    if missing:
        raise FileNotFoundError(f"poids manquants pour {name} : {missing}")
    with ThreadPoolExecutor(8) as ex:
        imgs = np.stack(list(tqdm(ex.map(lambda p: _load_resized(p, res), paths),
                                  total=len(paths), desc=f"{name} images")))
    model = timm.create_model(arch, pretrained=False, in_chans=4, num_classes=C.N_CLASSES)
    model = model.to(DEVICE).to(memory_format=torch.channels_last).eval()
    out = np.zeros((len(paths), C.N_CLASSES), np.float64)
    for w in weights:
        model.load_state_dict(torch.load(w, map_location=DEVICE))
        probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(paths), bs), desc=w.stem, leave=False):
                x = to_tensor(imgs[i:i + bs], DEVICE)
                with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                    p = model(x).float().softmax(1) + model(x.flip(3)).float().softmax(1)
                probs.append((p / 2).cpu().numpy())
        out += np.concatenate(probs) / len(weights)
    del model
    torch.cuda.empty_cache()
    return out


def knn_feats(emb_name, paths, df_all, y_train):
    """Features kNN des nouvelles images contre les embeddings du train."""
    ref = np.load(C.EMB_DIR / f"{emb_name}.npy")[(df_all.split == "train").to_numpy()]
    # float16 puis float32 : même arrondi que les embeddings stockés à l'entraînement
    q = embed(emb_name, list(paths)).astype(np.float16)
    f = _knn_block(torch.tensor(q).float().to(DEVICE), torch.tensor(ref).float().to(DEVICE), y_train)
    f.columns = [f"{emb_name}_{c}" for c in f.columns]
    torch.cuda.empty_cache()
    return f


def predict(images_dir, stack, save_probs=None):
    cfg = json.loads((C.MODELS_DIR / f"lgb_{stack}_config.json").read_text())
    df_all, tr, _ = load_index()
    paths = sorted(Path(images_dir).glob("*.png"))
    assert paths, f"aucun PNG dans {images_dir}"
    print(f"{len(paths)} images, stack {stack} : CNN {cfg['models']}, kNN {cfg['knn']}, "
          f"méta {cfg['meta']}, device {DEVICE}")

    with ProcessPoolExecutor() as ex:
        meta = pd.DataFrame(list(tqdm(ex.map(extract, map(str, paths), chunksize=64),
                                      total=len(paths), desc="métadonnées")))

    blocks, names = [], []
    for e in cfg["knn"]:
        f = knn_feats(e, paths, df_all, tr.y.to_numpy())
        blocks.append(f.to_numpy(np.float32))
        names += list(f.columns)
    for m in cfg["models"]:
        blocks.append(np.log(np.clip(cnn_probs(m, paths), 1e-6, 1)).astype(np.float32))
        names += [f"{m}_{c}" for c in C.CLASSES]
    if cfg["meta"]:
        cols = cfg["features"][len(names):]
        blocks.append(meta.reindex(columns=cols).to_numpy(np.float32))
        names += cols
    assert names == cfg["features"], "features incohérentes avec le stack entraîné"
    X = np.hstack(blocks)

    probs = np.zeros((len(paths), C.N_CLASSES))
    for k in range(C.N_FOLDS):
        bst = lgb.Booster(model_file=str(C.MODELS_DIR / f"lgb_{stack}_f{k}.txt"))
        probs += bst.predict(X) / C.N_FOLDS
    pred = probs.argmax(1)

    # Doublons exacts d'une image du train : on copie son label
    twin = meta.pixel_hash.map(tr.groupby("pixel_hash").y.first())
    dup = twin.notna().to_numpy()
    changed = int((pred[dup] != twin[dup].astype(int).to_numpy()).sum())
    pred[dup] = twin[dup].astype(int).to_numpy()
    print(f"doublons exacts du train : {dup.sum()} images, {changed} prédictions modifiées")

    if save_probs:
        np.save(save_probs, probs.astype(np.float32))
    ids = [int(p.stem) if p.stem.isdigit() else p.stem for p in paths]
    return pd.DataFrame({"Id": ids, "Label": np.array(C.CLASSES)[pred]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(C.TEST_DIR), help="dossier de PNG")
    ap.add_argument("--stack", default="deploy_B", choices=["deploy_A", "deploy_B"])
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--save-probs", default=None, help="chemin .npy pour les probabilités")
    args = ap.parse_args()
    sub = predict(args.images, args.stack, args.save_probs)
    sub.to_csv(args.out, index=False)
    print(sub.Label.value_counts(normalize=True).round(3).to_string())
    print(f"-> {args.out}")
