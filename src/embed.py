"""Embeddings préentraînés gelés (DINOv2, CLIP) pour train ∪ test -> outputs/embeddings/<nom>.npy

Les images sont composées sur fond blanc : on cherche à reconnaître le *concept* d'emoji
(même dessin chez plusieurs vendeurs), pas le style.
"""
import argparse

import numpy as np
import open_clip
import timm
import torch
from PIL import Image
from tqdm import tqdm

import config as C
from utils import load_index

MODELS = {
    "dinov2_s": ("timm", "vit_small_patch14_dinov2.lvd142m", 224),
    "dinov2_b": ("timm", "vit_base_patch14_dinov2.lvd142m", 224),
    "clip_b32": ("open_clip", ("ViT-B-32", "laion2b_s34b_b79k"), 224),
}


def load_rgb_white(path, res):
    with Image.open(path) as im:
        im = im.convert("RGBA").resize((res, res), Image.BICUBIC)
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


@torch.no_grad()
def embed(name, paths, bs=64):
    kind, spec, res = MODELS[name]
    if kind == "timm":
        model = timm.create_model(spec, pretrained=True, num_classes=0, img_size=res)
        cfg = timm.data.resolve_data_config({}, model=model)
        mean, std = cfg["mean"], cfg["std"]
        fwd = model
    else:
        model, _, pre = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1])
        mean, std = open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
        fwd = model.encode_image
    model = model.cuda().eval()
    mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    out = []
    for i in tqdm(range(0, len(paths), bs), desc=name):
        x = np.stack([load_rgb_white(p, res) for p in paths[i:i + bs]])
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            e = fwd((x - mean) / std).float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["dinov2_s", "clip_b32"])
    args = ap.parse_args()
    df, _, _ = load_index()
    for name in args.models:
        path = C.EMB_DIR / f"{name}.npy"
        if path.exists():
            print(f"{name} déjà calculé")
            continue
        e = embed(name, df.path.tolist())
        np.save(path, e.astype(np.float16))
        print(f"-> {path} {e.shape}")
