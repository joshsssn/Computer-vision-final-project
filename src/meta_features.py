"""Features de métadonnées PNG et statistiques d'image (sans réseau de neurones)."""
import hashlib
import struct

import numpy as np
from PIL import Image

PNG_SIG = b"\x89PNG\r\n\x1a\n"
KNOWN_CHUNKS = ["IHDR", "PLTE", "tRNS", "IDAT", "IEND", "pHYs", "gAMA", "cHRM",
                "sRGB", "iCCP", "tEXt", "zTXt", "iTXt", "bKGD", "tIME", "sBIT", "eXIf"]
MODES = ["RGBA", "RGB", "P", "L", "LA", "PA", "I", "1"]
HIST_BINS = 4  # 4x4x4 = 64 cases de couleur


def parse_png_chunks(raw: bytes) -> dict:
    """Lecture brute des chunks : IHDR, liste des chunks, IDAT, en-tête zlib."""
    f = {}
    if raw[:8] != PNG_SIG:
        f["png_valid"] = 0
        return f
    f["png_valid"] = 1
    pos, chunks, idat_sizes = 8, [], []
    zlib_hdr = None
    while pos + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8].decode("latin1")
        data = raw[pos + 8:pos + 8 + length]
        chunks.append(ctype)
        if ctype == "IHDR" and len(data) >= 13:
            _, _, bd, ct, _, _, il = struct.unpack(">IIBBBBB", data[:13])
            f.update(bit_depth=bd, color_type=ct, interlace=il)
        elif ctype == "IDAT":
            idat_sizes.append(length)
            if zlib_hdr is None and len(data) >= 2:
                zlib_hdr = data[:2]
        elif ctype == "PLTE":
            f["plte_entries"] = length // 3
        elif ctype == "tRNS":
            f["trns_len"] = length
        elif ctype == "pHYs" and len(data) >= 9:
            px, _, unit = struct.unpack(">IIB", data[:9])
            f.update(phys_x=px, phys_unit=unit)
        elif ctype == "gAMA" and len(data) >= 4:
            f["gamma_raw"] = struct.unpack(">I", data[:4])[0]
        pos += 12 + length
        if ctype == "IEND":
            break
    f["trailing_bytes"] = len(raw) - pos
    f["n_chunks"] = len(chunks)
    f["n_idat"] = len(idat_sizes)
    f["idat_total"] = sum(idat_sizes)
    f["idat_max"] = max(idat_sizes) if idat_sizes else 0
    if zlib_hdr is not None:
        f["zlib_cmf"], f["zlib_flg"] = zlib_hdr[0], zlib_hdr[1]
        f["zlib_level"] = zlib_hdr[1] >> 6
    for c in KNOWN_CHUNKS:
        f[f"chunk_{c}"] = chunks.count(c)
    f["chunk_other"] = sum(1 for c in chunks if c not in KNOWN_CHUNKS)
    f["chunk_order"] = "-".join(dict.fromkeys(chunks))
    return f


def pixel_hash(rgba: np.ndarray) -> str:
    return hashlib.md5(rgba.tobytes() + str(rgba.shape).encode()).hexdigest()


def image_stats(rgba: np.ndarray) -> dict:
    f = {}
    h, w = rgba.shape[:2]
    a = rgba[..., 3].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) / 255
    f["alpha_zero"] = float((a == 0).mean())
    f["alpha_full"] = float((a == 1).mean())
    f["alpha_semi"] = 1 - f["alpha_zero"] - f["alpha_full"]
    f["alpha_mean"] = float(a.mean())
    ys, xs = np.nonzero(a > 0)
    if len(ys):
        f["bbox_top"], f["bbox_bottom"] = ys.min() / h, (h - 1 - ys.max()) / h
        f["bbox_left"], f["bbox_right"] = xs.min() / w, (w - 1 - xs.max()) / w
    else:
        f["bbox_top"] = f["bbox_bottom"] = f["bbox_left"] = f["bbox_right"] = 1.0
    f["bbox_w"] = 1 - f["bbox_left"] - f["bbox_right"]
    f["bbox_h"] = 1 - f["bbox_top"] - f["bbox_bottom"]
    # Couleurs des pixels visibles
    vis = a > 0.5
    px = rgb[vis] if vis.any() else rgb.reshape(-1, 3)
    for i, c in enumerate("rgb"):
        f[f"mean_{c}"] = float(px[:, i].mean())
        f[f"std_{c}"] = float(px[:, i].std())
    mx, mn = px.max(1), px.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    f["sat_mean"], f["val_mean"] = float(sat.mean()), float(mx.mean())
    f["n_unique_colors"] = len(np.unique(np.ascontiguousarray(rgba).reshape(-1, 4).view(np.uint32)))
    f["unique_ratio"] = f["n_unique_colors"] / (h * w)
    # Anti-aliasing : fraction de pixels semi-transparents parmi les pixels non vides
    edge = (a > 0) & (a < 1)
    f["edge_ratio"] = float(edge.sum() / max((a > 0).sum(), 1))
    # Gradient moyen (netteté des traits)
    g = rgb.mean(2) * a
    f["grad_mean"] = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    # Contour sombre : fraction de pixels sombres parmi les visibles
    f["dark_ratio"] = float((px.max(1) < 0.2).mean())
    # Histogramme de couleurs quantifié
    q = np.clip((px * HIST_BINS).astype(int), 0, HIST_BINS - 1)
    idx = q[:, 0] * HIST_BINS ** 2 + q[:, 1] * HIST_BINS + q[:, 2]
    hist = np.bincount(idx, minlength=HIST_BINS ** 3) / len(idx)
    for i, v in enumerate(hist):
        f[f"hist_{i}"] = float(v)
    return f


def extract(path) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    f = {"file_size": len(raw)}
    f.update(parse_png_chunks(raw))
    with Image.open(path) as im:
        f["width"], f["height"] = im.size
        f["mode"] = im.mode
        f["format"] = im.format
        info = im.info
        f["info_keys"] = ",".join(sorted(info.keys()))
        f["has_dpi"] = int("dpi" in info)
        f["dpi"] = float(info["dpi"][0]) if "dpi" in info else -1.0
        f["has_transparency"] = int("transparency" in info)
        f["has_gamma"] = int("gamma" in info)
        f["has_icc"] = int("icc_profile" in info)
        f["icc_len"] = len(info.get("icc_profile", b"") or b"")
        f["palette_size"] = len(im.getpalette() or []) // 3 if im.mode == "P" else 0
        rgba = np.asarray(im.convert("RGBA"))
    f["bytes_per_pixel"] = f["file_size"] / (f["width"] * f["height"])
    f["aspect"] = f["width"] / f["height"]
    f["pixel_hash"] = pixel_hash(rgba)
    f.update(image_stats(rgba))
    for m in MODES:
        f[f"mode_{m}"] = int(f["mode"] == m)
    return f


# Colonnes non numériques ou identifiantes exclues des modèles
NON_FEATURE_COLS = {"id", "split", "path", "label", "y", "fold", "pixel_hash", "mode",
                    "format", "info_keys", "chunk_order", "dup_group", "concept"}


def feature_columns(df, pixels_only=False):
    """Colonnes numériques utilisables. pixels_only : seulement les stats calculées
    sur les pixels RGBA (sans taille native, chunks, mode, etc.)."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and df[c].dtype.kind in "biuf"]
    if pixels_only:
        pix = set(image_stats(np.zeros((4, 4, 4), np.uint8)).keys())
        cols = [c for c in cols if c in pix]
    return cols
