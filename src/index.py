"""Index train+test : métadonnées, hash pixel, folds -> outputs/index/index.parquet."""
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

import config as C
from meta_features import extract


def build_index() -> pd.DataFrame:
    labels = pd.read_csv(C.LABELS_CSV, dtype={"Id": str})
    rows = [dict(id=int(r.Id), split="train", path=str(C.TRAIN_DIR / f"{r.Id}.png"), label=r.Label)
            for r in labels.itertuples()]
    rows += [dict(id=int(p.stem), split="test", path=str(p), label=None)
             for p in sorted(C.TEST_DIR.glob("*.png"))]
    df = pd.DataFrame(rows)
    with ProcessPoolExecutor() as ex:
        feats = list(tqdm(ex.map(extract, df.path, chunksize=64), total=len(df), desc="features"))
    df = pd.concat([df, pd.DataFrame(feats)], axis=1)
    df["y"] = df.label.map(C.CLASS_TO_ID).fillna(-1).astype(int)
    df["dup_group"] = df.pixel_hash.astype("category").cat.codes

    # StratifiedGroupKFold, groupes = hash pixel : les doublons restent dans le même fold
    df["fold"] = -1
    tr = df.split == "train"
    skf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    for k, (_, va) in enumerate(skf.split(df[tr], df.y[tr], df.pixel_hash[tr])):
        df.loc[df.index[tr][va], "fold"] = k
    return df


def checks(df):
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    assert len(tr) == 9879 and len(te) == 9879
    assert (tr.y >= 0).all() and (tr.fold >= 0).all()
    assert tr.groupby("pixel_hash").fold.nunique().max() == 1, "fuite de doublons entre folds"
    conflicts = (tr.groupby("pixel_hash").y.nunique() > 1).sum()
    print(f"Folds sans fuite de hash. Conflits de label parmi les doublons train : {conflicts}")
    print(f"Images de test identiques à une image de train : {te.pixel_hash.isin(tr.pixel_hash).sum()}")
    print(tr.groupby("fold").label.value_counts().unstack())


if __name__ == "__main__":
    df = build_index()
    checks(df)
    df.to_parquet(C.INDEX_PATH)
    print(f"-> {C.INDEX_PATH}  ({df.shape[1]} colonnes)")
