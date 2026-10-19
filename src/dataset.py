"""Chargement RGBA, cache redimensionné (memmap uint8) et augmentations sur GPU.

Les augmentations préservent le style : petites translations/échelles, flip horizontal
optionnel, CutMix léger. Pas de jitter couleur (la palette est une signature du vendeur).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C

CACHE_DIR = C.OUT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(1, 4, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(1, 4, 1, 1)


def _load_resized(path, res):
    with Image.open(path) as im:
        im = im.convert("RGBA")
        if im.size != (res, res):
            im = im.resize((res, res), Image.BICUBIC)
        return np.asarray(im)


def get_cache(df_all, res):
    """Tableau (N, res, res, 4) uint8 aligné sur l'ordre de l'index complet (train puis test)."""
    path = CACHE_DIR / f"rgba_{res}.npy"
    if not path.exists():
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                        shape=(len(df_all), res, res, 4))
        with ThreadPoolExecutor(8) as ex:
            for i, img in enumerate(ex.map(lambda p: _load_resized(p, res), df_all.path)):
                arr[i] = img
        arr.flush()
        del arr
    return np.load(path, mmap_mode="r")


def to_tensor(batch_u8, device):
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN.to(device)) / STD.to(device)


def augment(x, hflip=True, max_shift=0.06, scale=(0.9, 1.1)):
    """Affine aléatoire (translation + échelle) et flip, fond transparent."""
    n = x.shape[0]
    dev = x.device
    s = torch.empty(n, device=dev).uniform_(*scale)
    tx = torch.empty(n, device=dev).uniform_(-max_shift, max_shift) * 2
    ty = torch.empty(n, device=dev).uniform_(-max_shift, max_shift) * 2
    sx = 1 / s
    if hflip:
        sx = sx * torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1] = sx, 1 / s
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # Fond = pixel transparent noir, exprimé dans l'espace normalisé
    bg = (-MEAN / STD).to(dev)
    return F.grid_sample(x - bg, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False) + bg


def cutmix(x, y_soft, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    n, _, h, w = x.shape
    perm = torch.randperm(n, device=x.device)
    rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, lam * y_soft + (1 - lam) * y_soft[perm]


class GpuBatches:
    """Itère sur des indices du cache et renvoie des tenseurs normalisés sur GPU."""

    def __init__(self, cache, idx, bs, device, shuffle, targets=None):
        self.cache, self.idx, self.bs = cache, np.asarray(idx), bs
        self.device, self.shuffle, self.targets = device, shuffle, targets

    def __len__(self):
        return (len(self.idx) + self.bs - 1) // self.bs

    def __iter__(self):
        order = np.random.permutation(len(self.idx)) if self.shuffle else np.arange(len(self.idx))
        with ThreadPoolExecutor(1) as ex:
            def load(j):
                sel = self.idx[order[j * self.bs:(j + 1) * self.bs]]
                srt = np.argsort(sel)  # lecture memmap ordonnée
                arr = self.cache[sel[srt]][np.argsort(srt)]
                t = None if self.targets is None else self.targets[order[j * self.bs:(j + 1) * self.bs]]
                return arr, t
            fut = ex.submit(load, 0)
            for j in range(len(self)):
                arr, t = fut.result()
                if j + 1 < len(self):
                    fut = ex.submit(load, j + 1)
                x = to_tensor(arr, self.device)
                yield (x, None if t is None else torch.as_tensor(t, device=self.device))
