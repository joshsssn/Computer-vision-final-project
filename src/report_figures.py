"""Figures du rapport (CPU uniquement) -> outputs/figures/

    python report_figures.py data       # exploration : classes, tailles natives, exemples, augmentations
    python report_figures.py results    # ablation (depuis outputs/ablation.csv)
"""
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import config as C
from utils import load_index

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
ORANGE, AQUA = "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": "#52514e", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    out = C.FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", out)


def fig_classes(tr):
    counts = tr.label.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 25, i, f"{v}", va="center", color="#52514e")
    ax.set_xlabel("training images")
    ax.set_title("Training images per vendor", loc="left", fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, counts.max() * 1.12)
    save(fig, "class_distribution.png")


def fig_sizes(tr):
    size = tr.width.astype(str) + "×" + tr.height.astype(str)
    top = size.value_counts().index[:5]
    size = size.where(size.isin(top), "other")
    tab = pd.crosstab(tr.label, size, normalize="index").reindex(C.CLASSES)
    tab = tab[[c for c in list(top) + ["other"] if c in tab.columns]]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.imshow(tab.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        color="white" if v > 0.55 else INK, fontsize=8)
    ax.set_title("Native image size by vendor (share of each vendor's images)", loc="left", fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "native_size_by_vendor.png")


def fig_samples(tr, n=8):
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(len(C.CLASSES), n, figsize=(n * 0.95, len(C.CLASSES) * 0.95))
    for r, c in enumerate(C.CLASSES):
        idx = rng.choice(np.flatnonzero(tr.label.to_numpy() == c), n, replace=False)
        for k, i in enumerate(idx):
            ax = axes[r, k]
            with Image.open(tr.path[i]) as im:
                ax.imshow(im.convert("RGBA"))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(c, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Random training samples per vendor", x=0.02, ha="left", fontsize=10)
    save(fig, "samples_per_vendor.png")


def fig_augment(tr, n_img=4, n_aug=6):
    from dataset import MEAN, STD, augment
    torch.manual_seed(0)
    rng = np.random.RandomState(3)
    idx = rng.choice(len(tr), n_img, replace=False)
    fig, axes = plt.subplots(n_img, n_aug + 1, figsize=((n_aug + 1) * 1.05, n_img * 1.05))
    for r, i in enumerate(idx):
        with Image.open(tr.path[i]) as im:
            a = np.asarray(im.convert("RGBA").resize((224, 224), Image.BICUBIC))
        x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float() / 255
        x = (x - MEAN) / STD
        xs = augment(x.repeat(n_aug, 1, 1, 1))
        for k in range(n_aug + 1):
            t = x[0] if k == 0 else xs[k - 1]
            img = (t * STD[0] + MEAN[0]).clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[r, k]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(GRID)
                s.set_visible(True)
    axes[0, 0].set_title("original", fontsize=8)
    axes[0, 1].set_title("augmented (shift, scale, flip) →", fontsize=8, loc="left")
    save(fig, "augmentations.png")


def fig_ablation():
    tab = pd.read_csv(C.OUT / "ablation_report.csv").iloc[::-1]
    col = {"visual": BLUE, "metadata": ORANGE, "pseudo": AQUA}
    lab = {"visual": "pixels only", "metadata": "uses PNG metadata",
           "pseudo": "uses test pseudo-labels (OOF optimistic)"}
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(tab) + 1.2))
    ax.barh(tab.config, tab.oof_acc * 100, color=[col[g] for g in tab.group], height=0.62)
    for i, v in enumerate(tab.oof_acc * 100):
        ax.text(v + 0.08, i, f"{v:.2f}", va="center", color="#52514e", fontsize=8)
    ax.set_xlim(84, 100)
    ax.set_xlabel("5-fold out-of-fold accuracy (%) — axis starts at 84%")
    ax.set_title("Ablation: contribution of each component", loc="left", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[g]) for g in col], labels=list(lab.values()),
              frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=3, fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    save(fig, "ablation.png")


def fig_confusion(name, title):
    from sklearn.metrics import confusion_matrix
    from utils import load_preds
    _, tr, _ = load_index()
    y = tr.y.to_numpy()
    cm = confusion_matrix(y, load_preds(name)[0].argmax(1))
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(C.N_CLASSES), C.CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(C.N_CLASSES), C.CLASSES)
    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, f"confusion_{name}_report.png")


def fig_recall():
    from utils import load_preds
    _, tr, _ = load_index()
    y = tr.y.to_numpy()
    models = [("cnx_t224", "ConvNeXt-T 224", "#9ec5f4"), ("stack_A", "Version A", BLUE),
              ("stack_B2", "Version B2", AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    w = 0.26
    for k, (m, lab, c) in enumerate(models):
        p = load_preds(m)[0].argmax(1)
        rec = [100 * (p[y == i] == i).mean() for i in range(C.N_CLASSES)]
        ax.bar(np.arange(C.N_CLASSES) + (k - 1) * w, rec, w * 0.92, color=c, label=lab)
    ax.set_xticks(range(C.N_CLASSES), C.CLASSES)
    ax.set_ylim(94, 100)
    ax.set_ylabel("OOF recall (%) — axis starts at 94%")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Per-class recall", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    save(fig, "recall_per_class.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "data"
    _, tr, _ = load_index()
    if what == "data":
        fig_classes(tr)
        fig_sizes(tr)
        fig_samples(tr)
        fig_augment(tr)
    else:
        fig_ablation()
        fig_confusion("stack_A", "Version A — out-of-fold confusion matrix (counts)")
        fig_confusion("stack_B2", "Version B2 — out-of-fold confusion matrix (counts)")
        fig_recall()
