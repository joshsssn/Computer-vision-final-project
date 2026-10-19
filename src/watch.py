"""Suivi en direct des entraînements (lecture seule des logs, n'interfère pas avec les runs).

    uv run python src/watch.py            # rafraîchit toutes les 10 s, Ctrl+C pour quitter
    uv run python src/watch.py --once
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import config as C

# File d'attente GPU (même ordre que run_night.sh) : nom, log, nombre de folds, époques
JOBS = [
    ("ConvNeXt-tiny 224", "log_cnx_t224.txt", 5, 30),
    ("ConvNeXt-tiny 160", "log_cnx_t160.txt", 5, 30),
    ("ConvNeXt 224 pseudo-labels", "log_cnx_t224_pl.txt", 5, 20),
    ("ConvNeXt 160 pseudo-labels", "log_cnx_t160_pl.txt", 5, 20),
    ("Ablation sans flip 160 (fold 0)", "log_abl_noflip.txt", 1, 30),
]
EP_RE = re.compile(r"fold (\d+) ep (\d+)/(\d+) loss ([\d.]+) \((\d+)s\)(?:\s+val_acc\(EMA\) ([\d.]+))?")
FOLD_RE = re.compile(r"fold (\d+) val_acc ([\d.]+)")
OOF_RE = re.compile(r"OOF accuracy = ([\d.]+)")
PREDICT_OVERHEAD = 120  # s de prédiction val+test par fold
DEFAULT_EP_S = {"224": 68, "160": 38}


def bar(frac, width=30):
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def fmt(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}" if sec >= 3600 else f"{sec // 60}min"


def parse(log, n_folds, n_ep):
    path = C.OUT / log
    if not path.exists() or path.stat().st_size == 0:
        return None
    txt = path.read_text(encoding="cp1252", errors="replace")
    eps = [m.groups() for m in EP_RE.finditer(txt)]
    done_folds = {int(k): float(a) for k, a in FOLD_RE.findall(txt)}
    oof = OOF_RE.search(txt)
    err = "Traceback" in txt or "out of memory" in txt
    return dict(eps=eps, done=done_folds, oof=float(oof.group(1)) if oof else None, err=err,
                mtime=path.stat().st_mtime)


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                              "temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        u, m, mt, t, clk = [x.strip() for x in out.split(",")]
        return f"GPU {u}% | VRAM {int(m) / 1024:.1f}/{int(mt) / 1024:.1f} Go | {t}°C | {clk} MHz"
    except Exception:
        return "GPU : nvidia-smi indisponible"


def render():
    now = time.time()
    lines = [f"=== Emoji vendor — suivi des entraînements — {datetime.now():%H:%M:%S} ===",
             gpu(), ""]
    t_cursor = now  # instant estimé de fin du job précédent
    for name, log, n_folds, n_ep in JOBS:
        st = parse(log, n_folds, n_ep)
        res = "224" if "224" in name else "160"
        if n_folds is None:  # job sans époques (embeddings)
            if st is None:
                eta = t_cursor + 600
                lines.append(f"[ en attente ] {name:<30} ~10min -> fin ~{datetime.fromtimestamp(eta):%H:%M}")
                t_cursor = eta
            elif (C.EMB_DIR / "knn_clip_b32_test.parquet").exists():
                lines.append(f"[   terminé  ] {name}")
            else:
                eta = max(t_cursor, now) + 300
                lines.append(f"[  en cours  ] {name:<30} -> fin ~{datetime.fromtimestamp(eta):%H:%M}")
                t_cursor = eta
            lines.append("")
            continue

        total = n_folds * n_ep
        if st is None or not st["eps"]:
            ep_s = DEFAULT_EP_S[res] * (2 if "pseudo" in name else 1)
            remaining = total * ep_s + n_folds * PREDICT_OVERHEAD
            eta = t_cursor + remaining
            state = "  démarrage " if st is not None else " en attente "
            lines.append(f"[{state}] {name:<30} {bar(0)}   0%  durée ~{fmt(remaining)} "
                         f"-> fin ~{datetime.fromtimestamp(eta):%H:%M}")
            t_cursor = eta
            lines.append("")
            continue

        k, e, _, loss, _, _ = st["eps"][-1]
        k, e = int(k), int(e)
        recent = [int(x[4]) for x in st["eps"][-5:]]
        ep_s = sorted(recent)[len(recent) // 2]
        done_ep = k * n_ep + e
        finished = st["oof"] is not None or len(st["done"]) == n_folds
        if st["err"]:
            state = "   ERREUR   "
        elif finished:
            state = "   terminé  "
        else:
            state = "  en cours  "
        # Progression interpolée dans l'époque en cours (le log n'est écrit qu'en fin d'époque)
        partial = 0.0 if finished else min((now - st["mtime"]) / ep_s, 0.99)
        frac = 1.0 if finished else (done_ep + partial) / total
        remaining = 0 if finished else \
            (total - done_ep - partial) * ep_s + (n_folds - k) * PREDICT_OVERHEAD
        eta = now + remaining
        t_cursor = eta if not finished else t_cursor
        last_val = next((x[5] for x in reversed(st["eps"]) if x[5]), None)
        lines.append(f"[{state}] {name:<30} {bar(frac)} {frac:4.0%}  "
                     + ("" if finished else f"reste {fmt(remaining)} -> fin ~{datetime.fromtimestamp(eta):%H:%M}"))
        lines.append(f"               fold {k + 1}/{n_folds}  époque {e}/{n_ep}  loss {loss}  "
                     f"{ep_s}s/époque" + (f"  val_acc {float(last_val):.4f}" if last_val else ""))
        if not finished:
            lines.append(f"               époque suivante {bar(partial, 20)} {partial:4.0%}"
                         f"  ({int(now - st['mtime'])}s / ~{ep_s}s)")
        if st["done"]:
            lines.append("               folds terminés : "
                         + "  ".join(f"f{f}={a:.4f}" for f, a in sorted(st["done"].items())))
        if st["oof"] is not None:
            lines.append(f"               OOF accuracy (5 folds) = {st['oof']:.4f}")
        if not finished and now - st["mtime"] > 10 * ep_s:
            lines.append(f"               /!\\ log inchangé depuis {fmt(now - st['mtime'])}")
        lines.append("")
    lines.append(f"Fin estimée de la file GPU : ~{datetime.fromtimestamp(t_cursor):%H:%M}"
                 f" (dans {fmt(t_cursor - now)})")
    lines.append("Meilleur LB : 0.98380 (sub03) | stacks : outputs/log_night.txt")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=0.5, help="secondes entre deux rafraîchissements")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.once:
        print(render())
    else:
        # Réécriture en place (curseur en haut, fin de ligne effacée) : pas de clignotement
        print("\033[2J\033[?25l", end="")
        try:
            while True:
                out = render()
                print("\033[H" + "\n".join(l + "\033[K" for l in out.split("\n")) + "\033[J",
                      end="", flush=True)
                time.sleep(args.every)
        except KeyboardInterrupt:
            pass
        finally:
            print("\033[?25h")
