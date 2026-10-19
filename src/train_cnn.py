"""Fine-tuning K-fold d'un CNN timm sur RGBA (4 canaux) -> OOF + probas test.

    python train_cnn.py --arch convnext_tiny.fb_in22k_ft_in1k --res 224 --name cnx_t224
    python train_cnn.py ... --pseudo stack_v1 --pseudo-thr 0.99   # avec pseudo-labels test
"""
import argparse
import math
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.utils import ModelEmaV3

import config as C
from dataset import GpuBatches, augment, cutmix, get_cache
from utils import load_index, load_preds, log_result, save_preds


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(C.N_FOLDS)))
    ap.add_argument("--no-hflip", action="store_true")
    ap.add_argument("--cutmix-p", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--pseudo", default=None, help="probas test utilisées comme pseudo-labels")
    ap.add_argument("--pseudo-thr", type=float, default=0.99)
    ap.add_argument("--max-train", type=int, default=0, help="debug : limite le train")
    return ap.parse_args()


@torch.no_grad()
def predict(model, cache, idx, bs, device):
    model.eval()
    out = []
    for x, _ in GpuBatches(cache, idx, bs * 2, device, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x).float().softmax(1) + model(x.flip(3)).float().softmax(1)
        out.append((p / 2).cpu().numpy())
    return np.concatenate(out)


def train_fold(args, k, cache, tr_idx, y_tr, va_idx, te_idx, class_w, device):
    torch.manual_seed(args.seed + k)
    np.random.seed(args.seed + k)
    model = timm.create_model(args.arch, pretrained=True, in_chans=4, num_classes=C.N_CLASSES,
                              drop_path_rate=args.drop_path).to(device)
    model = model.to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)
    ema = ModelEmaV3(model, decay=args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loader = GpuBatches(cache, tr_idx, args.bs, device, shuffle=True, targets=y_tr)
    total = args.epochs * len(loader)
    warm = max(1, min(len(loader), total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler()
    w = torch.tensor(class_w, device=device, dtype=torch.float32)

    # Reprise en cours de fold : checkpoint complet écrit à chaque époque
    ckpt_path = C.MODELS_DIR / f"{args.name}_f{k}_last.pt"
    start = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"]
        print(f"fold {k} : reprise à l'époque {start + 1}", flush=True)

    for ep in range(start, args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in loader:
            x = augment(x, hflip=not args.no_hflip).contiguous(memory_format=torch.channels_last)
            ys = F.one_hot(y.long(), C.N_CLASSES).float()
            ys = ys * (1 - args.smoothing) + args.smoothing / C.N_CLASSES
            if np.random.rand() < args.cutmix_p:
                x, ys = cutmix(x, ys)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            logp = logits.float().log_softmax(1)
            # Poids de classe modérés (racine de l'inverse de la fréquence)
            loss = (-(ys * logp) * w).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += loss.item() * len(x)
            n += len(x)
        msg = f"fold {k} ep {ep + 1}/{args.epochs} loss {tot / n:.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            pv = predict(ema.module, cache, va_idx[0], args.bs, device)
            msg += f"  val_acc(EMA) {(pv.argmax(1) == va_idx[1]).mean():.4f}"
        print(msg, flush=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep + 1},
                   tmp)
        tmp.replace(ckpt_path)  # écriture atomique : un crash pendant la sauvegarde ne corrompt rien

    pv = predict(ema.module, cache, va_idx[0], args.bs, device)
    pt = predict(ema.module, cache, te_idx, args.bs, device)
    torch.save(ema.module.state_dict(), C.MODELS_DIR / f"{args.name}_f{k}.pt")
    ckpt_path.unlink(missing_ok=True)
    return pv, pt


def main():
    args = parse()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    df, tr, te = load_index()
    cache = get_cache(df, args.res)
    # Position dans le cache = position dans l'index complet
    tr_pos = np.flatnonzero((df.split == "train").to_numpy())
    te_pos = np.flatnonzero((df.split == "test").to_numpy())
    y, folds = tr.y.to_numpy(), tr.fold.to_numpy()

    freq = np.bincount(y, minlength=C.N_CLASSES) / len(y)
    class_w = 1 / np.sqrt(freq)
    class_w = class_w / (class_w * freq).sum()

    pseudo_pos, pseudo_y = np.array([], int), np.array([], int)
    if args.pseudo:
        _, pt = load_preds(args.pseudo)
        keep = pt.max(1) >= args.pseudo_thr
        pseudo_pos, pseudo_y = te_pos[keep], pt[keep].argmax(1)
        print(f"pseudo-labels : {keep.sum()} images test (seuil {args.pseudo_thr})")

    part = C.OOF_DIR / "parts"
    part.mkdir(exist_ok=True)
    for k in args.folds:
        f_oof, f_te = part / f"{args.name}_f{k}_oof.npy", part / f"{args.name}_f{k}_test.npy"
        if f_oof.exists():
            print(f"fold {k} déjà fait")
            continue
        trn = np.flatnonzero(folds != k)
        if args.max_train:
            trn = np.random.RandomState(0).choice(trn, args.max_train, replace=False)
        val = np.flatnonzero(folds == k)
        tr_idx = np.concatenate([tr_pos[trn], pseudo_pos])
        y_tr = np.concatenate([y[trn], pseudo_y])
        pv, pt = train_fold(args, k, cache, tr_idx, y_tr, (tr_pos[val], y[val]), te_pos,
                            class_w, device)
        np.save(f_oof, pv)
        np.save(f_te, pt)
        print(f"fold {k} val_acc {(pv.argmax(1) == y[val]).mean():.4f}", flush=True)
        torch.cuda.empty_cache()

    if all((part / f"{args.name}_f{k}_oof.npy").exists() for k in range(C.N_FOLDS)):
        oof = np.zeros((len(tr), C.N_CLASSES))
        test = np.zeros((len(te), C.N_CLASSES))
        for k in range(C.N_FOLDS):
            oof[folds == k] = np.load(part / f"{args.name}_f{k}_oof.npy")
            test += np.load(part / f"{args.name}_f{k}_test.npy") / C.N_FOLDS
        save_preds(args.name, oof, test)
        log_result(args.name, y, oof, {"arch": args.arch, "res": args.res, "epochs": args.epochs,
                                       "hflip": not args.no_hflip, "pseudo": args.pseudo})


if __name__ == "__main__":
    main()
