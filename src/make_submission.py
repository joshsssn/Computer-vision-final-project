"""Soumission finale : probas test d'un modèle/stack -> règles transductives -> CSV Id,Label (noms de vendeurs).

    python make_submission.py --model stack_final --out sub_final.csv
    python make_submission.py --model meta_lgbm --no-dups --no-concepts
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from utils import load_index, load_preds


def copy_duplicates(tr, te, pred):
    """Label du jumeau train (pixels identiques) copié sur l'image de test."""
    hash_to_y = tr.groupby("pixel_hash").y.first()
    twin = te.pixel_hash.map(hash_to_y)
    mask = twin.notna().to_numpy()
    out = pred.copy()
    out[mask] = twin[mask].astype(int).to_numpy()
    changed = int((out != pred).sum())
    print(f"doublons exacts : {mask.sum()} images, {changed} prédictions modifiées")
    return out


def check_submission(sub, tr):
    assert list(sub.columns) == ["Id", "Label"]
    assert len(sub) == 9879, len(sub)
    assert sub.Id.tolist() == list(range(10001, 19880)), "IDs attendus 10001..19879"
    names = sub.Label.map(dict(enumerate(C.CLASSES))) if sub.Label.dtype.kind in "iu" else sub.Label
    assert names.isin(C.CLASSES).all()
    dist = pd.DataFrame({
        "train": tr.label.value_counts(normalize=True),
        "pred": names.value_counts(normalize=True)})
    print(dist.round(3))
    assert (dist.train - dist.pred).abs().max() < 0.03, "distribution prédite trop éloignée du train"
    print("contrôles de format : OK")


def make(model, use_dups=True, use_concepts=True, digits=False):
    _, tr, te = load_index()
    oof, test = load_preds(model)
    pred = test.argmax(1)
    if use_concepts:
        from concepts import apply_concepts
        pred = apply_concepts(tr, te, oof, test)
    if use_dups:
        pred = copy_duplicates(tr, te, pred)
    # Le leaderboard attend les noms de vendeurs (score 0 avec des chiffres)
    labels = pred.astype(int) if digits else np.array(C.CLASSES)[pred]
    sub = pd.DataFrame({"Id": te.id.astype(int), "Label": labels})
    sub = sub.sort_values("Id").reset_index(drop=True)
    check_submission(sub, tr)
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-dups", action="store_true")
    ap.add_argument("--no-concepts", action="store_true")
    ap.add_argument("--digits", action="store_true", help="labels 0..6 au lieu des noms")
    args = ap.parse_args()
    sub = make(args.model,
               use_dups=C.LEVERS["copy_duplicates"] and not args.no_dups,
               use_concepts=C.LEVERS["concept_constraint"] and not args.no_concepts,
               digits=args.digits)
    out = C.SUB_DIR / (args.out or f"sub_{args.model}.csv")
    sub.to_csv(out, index=False)
    print(f"-> {out}")
