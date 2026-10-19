"""Tableau d'ablation (accuracy OOF par levier), matrice de confusion, exemples d'erreurs.

    python ablation.py --cnn cnx_t224 cnx_t160 --knn dinov2_s clip_b32 --final stack_final
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

import config as C
from concepts import simulate_hungarian
from stack import build_features, run_lgb_cv
from utils import load_index, load_preds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def ablation_table(cnns, knn):
    _, tr, te = load_index()
    y, folds = tr.y.to_numpy(), tr.fold.to_numpy()
    rows = []

    def lgb_row(label, models=(), meta=False, pixels_only=False, knn_=(), visual=True):
        Xtr, Xte, _ = build_features(tr, te, list(models), meta, pixels_only, knn_)
        oof, _, _ = run_lgb_cv(Xtr, y, folds, Xte)
        rows.append(dict(config=label, oof_acc=acc(oof, y), visual_only=visual))
        print(rows[-1], flush=True)

    lgb_row("Métadonnées PNG + stats (LightGBM)", meta=True, visual=False)
    lgb_row("Stats de pixels seules (LightGBM)", meta=True, pixels_only=True)
    lgb_row("kNN DINOv2/CLIP seul", knn_=knn)
    lgb_row("Métadonnées + kNN", meta=True, knn_=knn, visual=False)
    for m in cnns:
        rows.append(dict(config=f"CNN {m} (brut)", oof_acc=acc(load_preds(m)[0], y), visual_only=True))
    if cnns:
        avg = np.mean([load_preds(m)[0] for m in cnns], 0)
        rows.append(dict(config="Moyenne des CNN", oof_acc=acc(avg, y), visual_only=True))
        lgb_row("Stack CNN (sans méta)", models=cnns)
        lgb_row("Stack CNN + kNN (visuel pur)", models=cnns, knn_=knn)
        lgb_row("Stack CNN + stats pixels + kNN (visuel pur)", models=cnns, meta=True,
                pixels_only=True, knn_=knn)
        lgb_row("Stack CNN + méta", models=cnns, meta=True, visual=False)
        lgb_row("Stack CNN + méta + kNN (complet)", models=cnns, meta=True, knn_=knn, visual=False)
    return pd.DataFrame(rows)


def confusion_fig(name):
    _, tr, _ = load_index()
    y = tr.y.to_numpy()
    p = load_preds(name)[0].argmax(1)
    cm = confusion_matrix(y, p, normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.N_CLASSES), C.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(C.N_CLASSES), C.CLASSES)
    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    ax.set_title(f"Matrice de confusion OOF — {name}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = C.FIG_DIR / f"confusion_{name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    raw = confusion_matrix(y, p)
    return out, pd.DataFrame(raw, index=C.CLASSES, columns=C.CLASSES)


def error_examples(name, n=24):
    _, tr, _ = load_index()
    y = tr.y.to_numpy()
    probs = load_preds(name)[0]
    p = probs.argmax(1)
    err = np.flatnonzero(p != y)
    err = err[np.argsort(-probs[err].max(1))][:n]  # erreurs les plus confiantes
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(len(err) / cols)), cols, figsize=(cols * 2, 2.3 * len(err) / cols))
    for ax, i in zip(axes.flat, err):
        with Image.open(tr.path[i]) as im:
            ax.imshow(im.convert("RGBA"))
        ax.set_title(f"true {C.CLASSES[y[i]]}\npred. {C.CLASSES[p[i]]} ({probs[i].max():.2f})\n"
                     f"{tr.width[i]}x{tr.height[i]} {tr['mode'][i]}", fontsize=7)
        ax.axis("off")
    for ax in list(axes.flat)[len(err):]:
        ax.axis("off")
    fig.tight_layout()
    out = C.FIG_DIR / f"errors_{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", nargs="*", default=[])
    ap.add_argument("--knn", nargs="*", default=["dinov2_s"])
    ap.add_argument("--final", default=None, help="modèle pour la matrice de confusion")
    args = ap.parse_args()

    tab = ablation_table(args.cnn, args.knn)
    if args.final:
        print("\nContrainte dure un-vendeur-par-concept (simulation OOF) :")
        for lo, hi in [(0.75, 0.93), (0.8, 0.9)]:
            d = simulate_hungarian(args.final, args.knn[0], lo, hi)
            tab.loc[len(tab)] = dict(config=f"{args.final} + contrainte dure {lo}-{hi}",
                                     oof_acc=float("nan"), visual_only=False, delta=d)
        out, raw = confusion_fig(args.final)
        print(raw)
        raw.to_csv(C.OUT / f"confusion_{args.final}.csv")
        print("->", out, error_examples(args.final))
    tab.to_csv(C.OUT / "ablation.csv", index=False)
    print(tab.to_string(index=False))
