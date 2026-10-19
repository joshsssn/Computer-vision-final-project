"""Pseudo-labelling : images test à confiance >= seuil -> réentraînement du CNN.

    python pseudo_label.py --source stack_v1 --thr 0.99 --name cnx_t224_pl1 -- --res 224

La précision attendue des pseudo-labels est estimée sur l'OOF du modèle source (précision des
exemples OOF au-dessus du seuil). Les doublons exacts train/test reçoivent le label train.
Remarque : les probas test du modèle source ont vu tout le train, donc l'OOF du modèle
réentraîné est légèrement optimiste (fuite indirecte via les pseudo-labels) ; on le signale
dans le rapport et on juge surtout sur le leaderboard pour ce levier.
"""
import argparse
import subprocess
import sys

import numpy as np

import config as C
from utils import load_index, load_preds, save_preds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="modèle/stack dont on prend les probas test")
    ap.add_argument("--thr", type=float, default=0.99)
    ap.add_argument("--name", required=True, help="nom du modèle réentraîné")
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    _, tr, te = load_index()
    oof, test = load_preds(args.source)
    y = tr.y.to_numpy()
    conf = oof.max(1) >= args.thr
    print(f"OOF : {conf.mean():.1%} au-dessus de {args.thr}, précision "
          f"{(oof[conf].argmax(1) == y[conf]).mean():.4f}")

    test = test.copy()
    twin = te.pixel_hash.map(tr.groupby("pixel_hash").y.first())
    m = twin.notna().to_numpy()
    test[m] = np.eye(C.N_CLASSES)[twin[m].astype(int)]
    print(f"test : {(test.max(1) >= args.thr).sum()} pseudo-labels (dont {m.sum()} doublons)")
    pl_name = f"pl_{args.source}"
    save_preds(pl_name, oof, test)

    extra = [a for a in args.train_args if a != "--"]
    cmd = [sys.executable, "-u", "train_cnn.py", "--name", args.name, "--pseudo", pl_name,
           "--pseudo-thr", str(args.thr)] + extra
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))
