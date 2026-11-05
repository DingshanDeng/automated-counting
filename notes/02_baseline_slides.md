# Baseline slides: Cellpose-SAM (2 slides, ~1 min)

> **Note 02** · Baseline (Cellpose-SAM) · Notebook: [`02_baseline_cellpose.ipynb`](../notebooks/02_baseline_cellpose.ipynb) · [All notes](README.md)

Numbers are from `notes/02_baseline_results.md` (75 test images, random split; leave-one-session-out on all 500).
The spoken script is about 150 words, which takes roughly 60 s at a normal pace.

---

## Slide 1: Baseline = off-the-shelf Cellpose-SAM, two thresholds tuned

**Title:** Baseline: a pretrained cell segmenter, no training

**On-slide text (keep it this short):**
- Cellpose-SAM (`cpsam_v2`): a published model that outlines every cell → we count the outlines
- Network weights are never trained on our images
- Only 3 numbers are tuned, on train images only:
  - cell-probability threshold 0 → **−4** (keep faint pixels)
  - flow (shape-check) threshold 0.4 → **1.0** (keep irregular outlines)
  - minimum area **400 px** (drop debris)

**Figure:** `figures/slides/slide1_defaults_vs_tuned.png` (black background, legend built in)
- Same image (`train/142`), defaults vs the tuned B3 setting (−4, 1.0, ≥ 400 px): 7 found vs 82 found, 82 labelled.
- Caption: *"Same image, same network. Defaults find 7 of 82 cells; tuned settings find about all of them."*

**Script (~30 s):**
> "Our baseline is Cellpose-SAM, a published cell-segmentation model. It outlines every cell, and we simply count the outlines. We never train it on our images. Out of the box, it's too strict for our pale, low-contrast cells: on this image it finds 7 of 82. So we tuned just two post-processing thresholds and a minimum cell size, using only the training images. Same network, and now it finds about every cell."

---

## Slide 2: Results: ~11% error with zero training

**Title:** Within 5.6 cells per image, and it generalizes to unseen sessions

**Figures** (pick one main figure; both are black-background):
- `figures/slides/slide2_pred_vs_true.png`: defaults vs tuned, predicted vs true count on the 75 test images, MAE in each title.
- `figures/slides/slide2_mae_unseen_session.png`: MAE bars for the session-average shortcut vs tuned Cellpose, on seen vs unseen sessions. It can replace the table below.

**Small table (on-slide):**

| | MAE (cells/img) | R² | Unseen session MAE |
|---|---|---|---|
| Session-mean shortcut (B1) | 6.9 | 0.71 | 14.6 |
| Cellpose, defaults (B2) | 34.8 | −6.2 | — |
| **Cellpose, tuned (B3)** | **5.6** | **0.74** | **6.3** |

**One-line takeaways (bottom of slide):**
- ✅ Unbiased (−0.6 cells), ~30× faster than counting by hand (1.85 s/image)
- ⚠️ Misses pale, faint cells; blur is *not* the main failure here
- ➡️ Next: trained models (YOLO, U-Net) must beat **MAE 5.6**

**Script (~30 s):**
> "Tuned, it counts within 5.6 cells per image, about 11% of a typical image, with R² 0.74. That beats a shortcut that just predicts each session's average, and, more importantly, on a session it has never seen, its error only rises to 6.3 while the shortcut doubles to 14.6. It takes under two seconds per image, about thirty times faster than counting by hand. Its main weakness is pale, faint cells. This is the bar our trained models have to beat."

---

## Tips

- If you only get **one slide**: use slide 2's B3 panel plus the table, and say slide 1's script in one sentence: *"pretrained Cellpose-SAM, two thresholds tuned on train, no training."*
- Likely Q&A:
  - *Why not fine-tune it?* We only have boxes, and Cellpose needs per-cell masks. That's a planned next step.
  - *Live vs Dead?* We count total cells for now. Viability is a later extension.
  - *Why trust the tuning?* It's a 15-pair grid search on the 351 training images only. Test images played no part.

## Figures

All figures are in `notes/figures/slides/` (200 dpi, black background, white text). They are regenerated from the cached Cellpose masks with `make_slide_figs.py` in the same folder (`conda run -n cellcount python notes/figures/slides/make_slide_figs.py`), so the model never runs again. The script needs the outputs of `notebooks/02_baseline_cellpose.ipynb` (the mask cache in `outputs/cache/cellpose/` and `outputs/results/baseline_cellpose.csv`), so run that notebook first.
