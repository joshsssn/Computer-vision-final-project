"""Leviers fondés sur les concepts d'emoji (embeddings gelés DINOv2 / CLIP).

Constat (DINOv2, voisin le plus proche dans le train) :
  sim >= 0.95 -> 98 % même vendeur (variantes de teinte de peau d'un même vendeur)
  sim 0.8-0.9 -> ~70-80 % vendeur différent (même concept, autre vendeur)
Les deux effets sont donc présents et de signes opposés.

1. Version « douce » : features kNN (similarité max par classe, votes pondérés) passées au
   stacking, qui apprend l'effet de chaque bande de similarité. Calcul sans fuite : un
   exemple du fold k n'a pour voisins que des exemples train des autres folds.
2. Version « dure » : composantes connexes (sim dans une bande) -> dans chaque groupe,
   vendeurs tous distincts et vendeurs train exclus, affectation hongroise sur les
   log-probas. Simulée sur l'OOF ; activée seulement si gain mesuré.

    python concepts.py knn --emb dinov2_s clip_b32     # features kNN -> outputs/embeddings
    python concepts.py hungarian --model stack_v1      # simulation OOF de la contrainte dure
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config as C
from utils import load_index, load_preds

K = 20
HI_SIM = 0.95


def load_emb(name):
    return torch.tensor(np.load(C.EMB_DIR / f"{name}.npy")).float().cuda()


def _knn_block(q, ref, ref_y, k=K):
    """Features kNN de q par rapport à ref (étiquetés)."""
    sims, idx = [], []
    for i in range(0, len(q), 2048):
        v, j = (q[i:i + 2048] @ ref.T).topk(k, dim=1)
        sims.append(v.cpu().numpy())
        idx.append(j.cpu().numpy())
    s, j = np.concatenate(sims), np.concatenate(idx)
    ny = ref_y[j]
    f = {"top1_sim": s[:, 0]}
    for c in range(C.N_CLASSES):
        m = ny == c
        f[f"maxsim_{c}"] = np.where(m, s, 0).max(1)
        f[f"vote_{c}"] = (np.where(m, np.exp((s - 1) * 20), 0)).sum(1)
        f[f"nhi_{c}"] = (m & (s >= HI_SIM)).sum(1)
        # Bande « même concept, autre vendeur »
        f[f"nmid_{c}"] = (m & (s >= 0.8) & (s < 0.93)).sum(1)
    tot = sum(f[f"vote_{c}"] for c in range(C.N_CLASSES)) + 1e-6
    for c in range(C.N_CLASSES):
        f[f"vote_{c}"] = f[f"vote_{c}"] / tot
    return pd.DataFrame(f)


def knn_features(name):
    """Features kNN sans fuite pour train (par fold) et test (voisins = tout le train)."""
    df, tr, te = load_index()
    e = load_emb(name)
    is_tr = (df.split == "train").to_numpy()
    etr, ete = e[np.flatnonzero(is_tr)], e[np.flatnonzero(~is_tr)]
    y, folds = tr.y.to_numpy(), tr.fold.to_numpy()
    ftr = None
    for k in range(C.N_FOLDS):
        ref = np.flatnonzero(folds != k)
        val = np.flatnonzero(folds == k)
        fk = _knn_block(etr[val], etr[ref], y[ref])
        fk.index = val
        ftr = fk if ftr is None else pd.concat([ftr, fk])
    ftr = ftr.sort_index()
    fte = _knn_block(ete, etr, y)
    ftr.columns = fte.columns = [f"{name}_{c}" for c in fte.columns]
    ftr.to_parquet(C.EMB_DIR / f"knn_{name}_train.parquet")
    fte.to_parquet(C.EMB_DIR / f"knn_{name}_test.parquet")
    acc = (ftr[[f"{name}_vote_{c}" for c in range(C.N_CLASSES)]].to_numpy().argmax(1) == y).mean()
    print(f"[{name}] kNN vote seul, accuracy OOF = {acc:.4f}")


def load_knn(names):
    if not names:
        return None, None
    ftr = pd.concat([pd.read_parquet(C.EMB_DIR / f"knn_{n}_train.parquet") for n in names], axis=1)
    fte = pd.concat([pd.read_parquet(C.EMB_DIR / f"knn_{n}_test.parquet") for n in names], axis=1)
    return ftr, fte


# ----------------------------------------------------------------- contrainte dure
def concept_groups(e, lo, hi, max_size=C.N_CLASSES):
    """Composantes connexes du graphe des paires de similarité dans [lo, hi)."""
    rows, cols = [], []
    for i in range(0, len(e), 2048):
        s = e[i:i + 2048] @ e.T
        r, c = torch.nonzero((s >= lo) & (s < hi), as_tuple=True)
        rows.append((r + i).cpu().numpy())
        cols.append(c.cpu().numpy())
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(e), len(e)))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    lab = np.where((sizes[lab] >= 2) & (sizes[lab] <= max_size), lab, -1)
    return lab


def assign(groups, known_y, logp):
    """known_y : label connu (>=0) ou -1. Renvoie la prédiction après contrainte."""
    pred = logp.argmax(1).copy()
    order = np.argsort(groups)
    g_sorted = groups[order]
    bounds = np.flatnonzero(np.diff(g_sorted)) + 1
    for members in np.split(order, bounds):
        if groups[members[0]] < 0:
            continue
        unk = members[known_y[members] < 0]
        used = set(known_y[members][known_y[members] >= 0])
        if len(unk) == 0 or len(used) != (known_y[members] >= 0).sum():
            continue  # groupe incohérent (même vendeur deux fois côté train) -> on ignore
        free = [c for c in range(C.N_CLASSES) if c not in used]
        if len(unk) > len(free):
            continue
        cost = -logp[np.ix_(unk, free)]
        ri, ci = linear_sum_assignment(cost)
        pred[unk[ri]] = np.array(free)[ci]
    return pred


def simulate_hungarian(model, emb, lo, hi):
    """Pour chaque fold : connus = train hors fold, inconnus = fold + test."""
    df, tr, te = load_index()
    oof, test = load_preds(model)
    e = load_emb(emb)
    groups = concept_groups(e, lo, hi)
    y, folds = tr.y.to_numpy(), tr.fold.to_numpy()
    ntr = len(tr)
    logp_all = np.log(np.clip(np.vstack([oof, test]), 1e-9, 1))
    new = oof.argmax(1).copy()
    for k in range(C.N_FOLDS):
        known = np.full(len(df), -1)
        known[:ntr] = np.where(folds != k, y, -1)
        p = assign(groups, known, logp_all)
        new[folds == k] = p[:ntr][folds == k]
    base = (oof.argmax(1) == y).mean()
    after = (new == y).mean()
    in_grp = (groups[:ntr] >= 0).mean()
    print(f"[{model} | {emb} {lo}-{hi}] OOF {base:.4f} -> {after:.4f} "
          f"({after - base:+.4f}), {in_grp:.1%} du train dans un groupe")
    return after - base


def apply_concepts(tr, te, oof, test, emb="dinov2_s", lo=0.8, hi=0.93):
    """Contrainte dure appliquée au test (connus = tout le train)."""
    e = load_emb(emb)
    groups = concept_groups(e, lo, hi)
    known = np.concatenate([tr.y.to_numpy(), np.full(len(te), -1)])
    logp = np.log(np.clip(np.vstack([oof, test]), 1e-9, 1))
    pred = assign(groups, known, logp)[len(tr):]
    print(f"contrainte concept : {(pred != test.argmax(1)).sum()} prédictions test modifiées")
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["knn", "hungarian"])
    ap.add_argument("--emb", nargs="*", default=["dinov2_s", "clip_b32"])
    ap.add_argument("--model", default="meta_lgbm")
    args = ap.parse_args()
    if args.cmd == "knn":
        for n in args.emb:
            knn_features(n)
    else:
        for n in args.emb:
            for lo, hi in [(0.75, 0.93), (0.8, 0.93), (0.85, 0.95), (0.8, 0.9)]:
                simulate_hungarian(args.model, n, lo, hi)
