# Plan

One row per weekly check-in. Scope, data and KPIs are in [`project.md`](project.md).

| Week | Check-in | What we deliver | Status |
|---|---|---|---|
| 1 | Scope, data sources, stakeholders, KPIs | [`project.md`](project.md) | ✅ Done |
| 2 | EDA and preprocessing | [`01_eda_overview.ipynb`](../notebooks/01_eda_overview.ipynb), the shared split [`splits.csv`](../splits/splits.csv), [`data.md`](data.md) | ✅ Done |
| 3 | Model architecture: baseline and deep learning | Models numbered in the order we test them: B0/B1 references, B2/B3 Cellpose-SAM (default / tuned on train, the baseline), then B4 YOLO and B5 U-Net trained on our boxes ([`baseline_explained.md`](baseline_explained.md)) | ✅ Done |
| 4 | Baseline performance | [`02_baseline_cellpose.ipynb`](../notebooks/02_baseline_cellpose.ipynb), [`baseline_results.md`](baseline_results.md) | ✅ First run |
| 5 | Deep-learning model, iteration 1 | B4 YOLO26n trained on Colab with default settings, scored like the baseline | Next |
| 6 | Deep-learning model, iteration 2 | B5 U-Net density map; YOLO improved one change at a time (augmentation, model size, leave-one-session-out) | Planned |
| 7 | Final project | Clean repo, 1-page executive summary, 5-minute presentation | Planned |

## How we evaluate

- **Data:** the 500 Roboflow images, split by [`splits/splits.csv`](../splits/splits.csv) (351 train / 74 valid / 75 test).
- **Task for now:** the total number of cells per image. Telling Live from Dead comes later.
- **Scores:** total-count MAE (cells per image), ±10% accuracy, R², and seconds per image. For models that find cells, we also report precision and recall.
- **Fair settings:** anything we tune is chosen on train (or valid), never on test.
- **Two test levels:** the random split, and leave-one-session-out (train on 4 sessions, test on the 5th). The session alone predicts the count with R² ≈ 0.7, so a model has to beat that.
