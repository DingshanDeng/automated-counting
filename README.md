# fall-2026-automated-cell-counting
Team project: fall-2026-automated-cell-counting

A deep-learning tool that counts cells in microscopy images. Scope and KPIs are in [`notes/project.md`](notes/project.md), and the weekly roadmap is in [`notes/plan.md`](notes/plan.md). All write-ups are listed in the [notes index](notes/README.md).

## Start here

1. **Set up** (local): `conda env create -f environment.yml`, then copy `.env.example` to `.env` and fill in `ROBOFLOW_EXPORT_URL`. On Colab, add the same name under Secrets. The first cell of each notebook installs the package.
2. **Run the notebooks in order:**

| Notebook | What it does | Notes |
|---|---|---|
| [`00_download_data.ipynb`](notebooks/00_download_data.ipynb) | Downloads the 500 Roboflow images into `data/` | [`notes/00_data.md`](notes/00_data.md) |
| [`01_eda_overview.ipynb`](notebooks/01_eda_overview.ipynb) | Exploratory data analysis: counts, Live/Dead, image quality, overlap, colour | [`01_eda`](notes/01_eda.md) |
| [`02_baseline_cellpose.ipynb`](notebooks/02_baseline_cellpose.ipynb) | Baseline: off-the-shelf Cellpose-SAM counts cells (GPU recommended) | [`02_baseline_results`](notes/02_baseline_results.md), [`02_baseline_explained`](notes/02_baseline_explained.md) |
| [`03_initial_model_train.ipynb`](notebooks/03_initial_model_train.ipynb) | Our model: YOLO26n trained on our images | [`03_yolo_training`](notes/03_yolo_training.md) |
| [`03a_rgb_grayscale.ipynb`](notebooks/03a_rgb_grayscale.ipynb) | YOLO26n trained on RGB vs grayscale input | [`03a_rgb_grayscale_results`](notes/03a_rgb_grayscale_results.md), [`03a_rgb_grayscale_diagnostics`](notes/03a_rgb_grayscale_diagnostics.md) |
| [`03b_augmentation_exploration.ipynb`](notebooks/03b_augmentation_exploration.ipynb) | Screens training augmentations one at a time | [`03_yolo_training`](notes/03_yolo_training.md) |
| [`04_yolo_performance.ipynb`](notebooks/04_yolo_performance.ipynb) | Our model vs the baseline on the shared test split | [`04_yolo_performance`](notes/04_yolo_performance.md) |
| [`05a_model_application.ipynb`](notebooks/05a_model_application.ipynb) | Applies our model to synthetic cell images (BBBC005) | [`05_extra_experiments`](notes/05_extra_experiments.md) |
| [`05b_extra_experimentation.ipynb`](notebooks/05b_extra_experimentation.ipynb) | Applies our model to other datasets (blood cells, tomatoes, eggs) | [`05_extra_experiments`](notes/05_extra_experiments.md) |

## Layout

```
notebooks/        notebooks that run the analysis
src/cellcount/    reusable code (data loading, EDA, splits, metrics, models)
splits/           splits.csv: the shared train/valid/test split, which every model must use
notes/            write-ups numbered like their notebooks; start at notes/README.md
                  (slide figures: notes/figures/slides/, the note figures in the slide theme)
data/, outputs/   gitignored: downloaded images, results, model caches
```
