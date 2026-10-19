#!/usr/bin/env bash
# Enchaînement complet (un seul job GPU à la fois). Chaque étape est reprenable :
# relancer ce script saute ce qui est déjà fait.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
O=../outputs
ARCH=convnext_tiny.fb_in22k_ft_in1k
KNN="dinov2_s clip_b32"
step() { echo "=== $(date +%H:%M) $* ===" >> $O/log_night.txt; }
cpu() { CUDA_VISIBLE_DEVICES="" uv run python -u "$@" >> $O/log_night.txt 2>&1; }

# 1. ConvNeXt 160 (reprise du checkpoint) -------------------------------------------
step "cnx_t160"
[ -f $O/oof/cnx_t160.npy ] || uv run python -u train_cnn.py --arch $ARCH --res 160 --epochs 30 --seed 7 --name cnx_t160 >> $O/log_cnx_t160.txt 2>&1

# Version A : sans pseudo-labels
step "stack_A (sans pseudo-labels)"
cpu stack.py --name stack_A --models cnx_t224 cnx_t160 --knn $KNN
cpu stack.py --name stack_A_visual --models cnx_t224 cnx_t160 --knn $KNN --pixels-only
cpu make_submission.py --model stack_A --out final_A_no_pseudo.csv

# 2. Pseudo-labels depuis stack_A -> ConvNeXt 224 --------------------------------------
step "cnx_t224_pl"
[ -f $O/oof/cnx_t224_pl.npy ] || uv run python -u pseudo_label.py --source stack_A --thr 0.99 --name cnx_t224_pl -- --arch $ARCH --res 224 --epochs 20 --seed 11 >> $O/log_cnx_t224_pl.txt 2>&1

step "stack_B1 (avec pseudo-labels, 224)"
cpu stack.py --name stack_B1 --models cnx_t224 cnx_t160 cnx_t224_pl --knn $KNN
cpu make_submission.py --model stack_B1 --out final_B1_pseudo.csv

# 3. Pseudo-labels -> ConvNeXt 160 ---------------------------------------------------
step "cnx_t160_pl"
[ -f $O/oof/cnx_t160_pl.npy ] || uv run python -u pseudo_label.py --source stack_A --thr 0.99 --name cnx_t160_pl -- --arch $ARCH --res 160 --epochs 20 --seed 13 >> $O/log_cnx_t160_pl.txt 2>&1

step "stack_B2 (avec pseudo-labels, 224 + 160)"
cpu stack.py --name stack_B2 --models cnx_t224 cnx_t160 cnx_t224_pl cnx_t160_pl --knn $KNN
cpu make_submission.py --model stack_B2 --out final_B2_pseudo.csv

# 4. Ablation flip (fold 0, 160 px) --------------------------------------------------
step "ablation sans flip"
[ -f $O/oof/parts/abl_noflip_t160_f0_oof.npy ] || uv run python -u train_cnn.py --arch $ARCH --res 160 --epochs 30 --seed 7 --folds 0 --no-hflip --name abl_noflip_t160 >> $O/log_abl_noflip.txt 2>&1

step "NIGHT_DONE"
