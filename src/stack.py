"""Stacking LightGBM : [probas OOF des modèles] + [métadonnées], mêmes folds que les modèles.

Sans modèle en entrée, c'est la baseline « métadonnées + histogrammes ».

    python stack.py --name meta_lgbm                       # baseline
    python stack.py --name stack --models cnn_a cnn_b      # stacking
    python stack.py --name stack_nometa --models cnn_a --no-meta
    python stack.py --name meta_knn --knn dinov2_s clip_b32
"""
import argparse
import json

import lightgbm as lgb
import numpy as np

import config as C
from concepts import load_knn
from meta_features import feature_columns
from utils import load_index, load_preds, log_result, save_preds

LGB_PARAMS = dict(objective="multiclass", num_class=C.N_CLASSES, learning_rate=0.03,
                  num_leaves=31, min_child_samples=20, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  seed=C.SEED, num_threads=8)


def build_features(tr, te, models, use_meta, pixels_only=False, knn=()):
    Xtr, Xte, names = [], [], []
    ktr, kte = load_knn(list(knn))
    if ktr is not None:
        Xtr.append(ktr.to_numpy(np.float32))
        Xte.append(kte.to_numpy(np.float32))
        names += list(ktr.columns)
    for m in models:
        o, t = load_preds(m)
        Xtr.append(np.log(np.clip(o, 1e-6, 1)))
        Xte.append(np.log(np.clip(t, 1e-6, 1)))
        names += [f"{m}_{c}" for c in C.CLASSES]
    if use_meta:
        cols = feature_columns(tr, pixels_only=pixels_only)
        Xtr.append(tr[cols].to_numpy(np.float32))
        Xte.append(te[cols].to_numpy(np.float32))
        names += cols
    return np.hstack(Xtr), np.hstack(Xte), names


def run_lgb_cv(Xtr, y, folds, Xte, params=LGB_PARAMS, rounds=3000, save_as=None, names=None):
    oof = np.zeros((len(y), C.N_CLASSES))
    test = np.zeros((len(Xte), C.N_CLASSES))
    iters = []
    for k in range(C.N_FOLDS):
        trn, val = folds != k, folds == k
        dtr = lgb.Dataset(Xtr[trn], y[trn])
        dva = lgb.Dataset(Xtr[val], y[val])
        # L'arrêt précoce sur le fold de validation biaise légèrement l'OOF vers le haut ;
        # on le garde pour la baseline, le nombre d'itérations médian sert au modèle final.
        bst = lgb.train(params, dtr, rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val] = bst.predict(Xtr[val], num_iteration=bst.best_iteration)
        test += bst.predict(Xte, num_iteration=bst.best_iteration) / C.N_FOLDS
        iters.append(bst.best_iteration)
        if save_as:  # modèles gardés pour l'inférence sans réentraînement (predict.py)
            bst.save_model(str(C.MODELS_DIR / f"lgb_{save_as}_f{k}.txt"),
                           num_iteration=bst.best_iteration)
    if save_as and names is not None:
        (C.MODELS_DIR / f"lgb_{save_as}_features.json").write_text(json.dumps(names))
    return oof, test, iters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="meta_lgbm")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--knn", nargs="*", default=[], help="features kNN d'embeddings (concepts.py)")
    ap.add_argument("--pixels-only", action="store_true",
                    help="seulement les stats calculées sur les pixels (pas de métadonnées PNG)")
    args = ap.parse_args()

    _, tr, te = load_index()
    Xtr, Xte, names = build_features(tr, te, args.models, not args.no_meta, args.pixels_only,
                                     args.knn)
    y, folds = tr.y.to_numpy(), tr.fold.to_numpy()
    print(f"{Xtr.shape[1]} features, modèles : {args.models or '-'}")
    oof, test, iters = run_lgb_cv(Xtr, y, folds, Xte, save_as=args.name, names=names)
    # Recette du stack, relue par predict.py pour reconstruire les mêmes features
    (C.MODELS_DIR / f"lgb_{args.name}_config.json").write_text(json.dumps(
        {"models": args.models, "knn": args.knn, "meta": not args.no_meta,
         "pixels_only": args.pixels_only, "features": names}))
    print("itérations :", iters)
    save_preds(args.name, oof, test)
    log_result(args.name, y, oof, {"models": args.models, "meta": not args.no_meta,
                                   "pixels_only": args.pixels_only, "knn": args.knn})
