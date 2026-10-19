"""Fonctions partagées : index, sauvegarde des prédictions, journal des scores."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import config as C

RESULTS_LOG = C.OUT / "results.jsonl"


def load_index():
    df = pd.read_parquet(C.INDEX_PATH)
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    return df, tr, te


def save_preds(name, oof, test):
    np.save(C.OOF_DIR / f"{name}.npy", oof.astype(np.float32))
    np.save(C.TEST_PROBS_DIR / f"{name}.npy", test.astype(np.float32))


def load_preds(name):
    return np.load(C.OOF_DIR / f"{name}.npy"), np.load(C.TEST_PROBS_DIR / f"{name}.npy")


def per_class_recall(y, pred):
    return {c: float((pred[y == i] == i).mean()) for i, c in enumerate(C.CLASSES)}


def log_result(name, y, oof_probs, extra=None):
    pred = oof_probs.argmax(1) if oof_probs.ndim == 2 else oof_probs
    acc = accuracy_score(y, pred)
    rec = per_class_recall(y, pred)
    print(f"[{name}] OOF accuracy = {acc:.4f}")
    print("  rappel : " + "  ".join(f"{c}={v:.3f}" for c, v in rec.items()))
    rec_ = {"name": name, "oof_acc": acc, "recall": rec, "time": time.strftime("%Y-%m-%d %H:%M")}
    rec_.update(extra or {})
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec_) + "\n")
    return acc
