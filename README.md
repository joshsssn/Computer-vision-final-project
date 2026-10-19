# Emoji Vendor Classification

Predict which vendor designed an emoji image — **apple, google, whatsapp, facebook, samsung,
mozilla or messenger** — for a course computer-vision challenge. 9,879 labelled training images,
9,879 test images, metric = accuracy.

**Authors:** Josh SOUSSAN, Edouard SOUIED

## Results

| Model | 5-fold OOF | Public leaderboard |
|---|---|---|
| LightGBM on PNG metadata + image stats + DINOv2 kNN features (sub02) | 0.9651 | 0.96618 |
| ConvNeXt-tiny 224, RGBA input, fold 0 only (sub04) | 0.9828 (fold 0) | 0.98238 |
| 0.7 x ConvNeXt fold 0 + 0.3 x metadata/kNN model (sub03) | 0.9853 (fold 0) | 0.98380 |
| Pixels-only stack: ConvNeXt 224 + kNN + pixel statistics, no PNG metadata (sub07) | 0.9822 | 0.98582 |
| Stack: ConvNeXt 224 + DINOv2 kNN + metadata (sub06) | 0.9832 | 0.98582 |
| Stack: ConvNeXt 224 + DINOv2/CLIP kNN + metadata (sub05) | 0.9834 | 0.98623 |
| Pixels-only stack: 2 CNNs + kNN + pixel statistics | 0.9829 | - |
| **Version A** - stack: ConvNeXt 224 + 160 + kNN + metadata, no pseudo-labels | 0.9830 | **0.98663** |
| **Version B1** - Version A + pseudo-labelled ConvNeXt 224 | 0.9845 (optimistic) | 0.98643 |
| **Version B2** - Version A + both pseudo-labelled ConvNeXt | 0.9847 (optimistic) | 0.98582 |

Final files are in `outputs/submissions/`: `final_A_no_pseudo.csv`, `final_B1_pseudo.csv`,
`final_B2_pseudo.csv`. The three final scores are within four images of each other on the
public half of the test set: pseudo-labelling raised out-of-fold (OOF) accuracy but not the
leaderboard score. Two further leaderboard results: removing PNG metadata changes nothing
(sub07 and sub06 both score 0.98582), and the single-CNN stack (sub05) is only two images
behind Version A. The full history is in
[`outputs/submissions/leaderboard.csv`](outputs/submissions/leaderboard.csv), every run is
logged to `outputs/results.jsonl`, and the written report is
[`report/report.pdf`](report/report.pdf) (source: `report/report.tex`).

## Predicting without retraining

`predict.py` goes from a folder of PNG images to a CSV using only saved weights.

**1. Download the CNN weights** (1.67 GB, too large for git) from
[Google Drive](https://drive.google.com/file/d/1C5Hchff3VvjfUGzjUUFHWyAZP3j58kiN/view?usp=sharing),
or from the command line, at the project root:

```bash
uv run --with gdown gdown 1C5Hchff3VvjfUGzjUUFHWyAZP3j58kiN -O model_weights.zip
```

**2. Unzip it at the project root.** The archive already contains the `outputs/models/` paths:

```bash
uv run python -m zipfile -e model_weights.zip .
```

**3. Predict:**

```bash
uv run python src/predict.py --images test --stack deploy_B --out predictions.csv
```

| Stack | Content | OOF |
|---|---|---|
| `deploy_B` (default) | ConvNeXt 160 + pseudo-labelled ConvNeXt 224 and 160 + kNN + metadata | 0.9842 |
| `deploy_A` | ConvNeXt 160 + kNN + metadata, no pseudo-labels | 0.9805 |

Besides the downloaded weights, it uses files already in the repository: the LightGBM models
(`outputs/models/lgb_deploy_*`), `outputs/index/index.parquet` and `outputs/embeddings/*.npy`
(the kNN reference set). It takes about 11 minutes for the 9,879 test images on the laptop GPU. Run on `test/`, it
reproduces the stored `deploy_B` predictions exactly. These two stacks differ from Versions A
and B because the weights of the first ConvNeXt 224 run were not saved, only its predictions;
`deploy_B` disagrees with Version A on 45 of 9,879 test predictions.

## Approach

1. **Metadata index** — native size, PNG mode, raw chunk layout, zlib header, palette, alpha
   coverage, content bounding box, colour histogram, and an MD5 hash of the RGBA pixels.
   Native size alone separates vendor groups (240 px → facebook/messenger/mozilla,
   160 px → whatsapp, 72 px → apple/google/samsung).
2. **Validation** — 5-fold `StratifiedGroupKFold` grouped by pixel hash, so exact duplicates
   never straddle folds. The same folds are shared by every model, which keeps stacking valid.
3. **CNNs** — timm ConvNeXt fine-tuned on **4-channel RGBA** input, AdamW + cosine schedule,
   EMA, label smoothing, square-root inverse-frequency class weights, flip TTA.
   Augmentations preserve style (small shifts/scales, horizontal flip, light CutMix) and
   deliberately avoid colour jitter, since the palette is a vendor signature.
4. **Frozen embeddings** — DINOv2 / CLIP embeddings of the emoji composited on white, turned
   into leak-free kNN features (per-class max similarity, weighted votes, similarity bands).
5. **Stacking** — LightGBM on CNN log-probabilities + metadata + kNN features, same folds.
6. **Transductive rules** — test images identical to a training image get its label
   (349 images, no label conflicts among training duplicates). A hard
   "one vendor per emoji concept" Hungarian assignment was simulated on OOF and **rejected**
   (−0.6 to −2.1 pt): near-identical skin-tone variants from the same vendor contaminate the
   concept groups.
7. **Pseudo-labelling (Version B)** — retrain the CNNs with the 9,046 test images that
   Version A predicts at ≥ 0.99 confidence, then stack again.

### Use of metadata

The brief says vendors should be inferred "based solely on visual characteristics". PNG
metadata (chunk layout, native size, palette mode) is highly predictive, so the report gives
a pure-visual score (CNN + pixel statistics + visual kNN) next to the full pipeline score.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (developed on an RTX 3050 Ti,
4 GB). PyTorch is pulled from the CUDA 12.8 wheel index configured in `pyproject.toml`.

```bash
uv sync
```

Place the challenge data at the project root:

```
train/00001.png … train/09879.png
test/10001.png  … test/19879.png
train_labels.csv
```

## Running the pipeline

All scripts live in `src/` and are run from there with `uv run python <script>`.

```bash
cd src
uv run python index.py                                   # metadata + pixel hashes + folds
uv run python embed.py --models dinov2_s clip_b32        # frozen embeddings
uv run python concepts.py knn --emb dinov2_s clip_b32    # kNN features
uv run python train_cnn.py --arch convnext_tiny.fb_in22k_ft_in1k --res 224 --name cnx_t224
bash run_night.sh        # ConvNeXt 160 -> Version A -> pseudo-labelled CNNs -> Version B
uv run python ablation.py --cnn cnx_t224 cnx_t160 --knn dinov2_s clip_b32 --final stack_A
uv run python report_figures.py data && uv run python report_figures.py results
```

`run_night.sh` chains the remaining GPU jobs one at a time and writes the stack scores to
`outputs/log_night.txt`; rerunning it skips whatever is already done.

Training is resumable. Finished folds are skipped on relaunch, and runs write a checkpoint
each epoch so an interrupted fold continues where it stopped. To follow training live:

```bash
uv run python src/watch.py            # refreshes every 0.5 s, Ctrl+C to quit
```

Submissions are written to `outputs/submissions/` as `Id,Label` with **vendor names** as
labels (the grader scores numeric labels as 0). `make_submission.py` checks row count, ids,
label set and predicted class distribution before writing.

## Repository layout

```
src/
  config.py          paths, classes, seeds, lever switches
  meta_features.py   PNG chunk parsing and image statistics
  index.py           train+test index, pixel hashes, folds
  dataset.py         resized RGBA cache (memmap) and GPU augmentations
  train_cnn.py       K-fold fine-tuning -> OOF + test probabilities
  embed.py           frozen DINOv2 / CLIP embeddings
  concepts.py        kNN features and the Hungarian concept-constraint experiment
  stack.py           LightGBM stacking (also the metadata baseline)
  pseudo_label.py    retraining with confident test pseudo-labels
  make_submission.py transductive rules, CSV writing and checks
  predict.py         inference from saved weights: images folder -> CSV
  ablation.py        ablation table, confusion matrix, error examples
  watch.py           live training dashboard
  report_figures.py  figures used in the report
  run_night.sh       sequential GPU job queue (Version A, then Version B)
outputs/
  oof/ test_probs/   per-model probabilities (oof/parts/ holds per-fold files)
  submissions/       CSVs and leaderboard.csv
  figures/           report figures
report/              report.tex and report.pdf
```
