# fall-2026-automated-cell-counting
Team project: fall-2026-automated-cell-counting

A deep-learning tool that counts cells in microscopy images. Scope and KPIs are in [`notes/project.md`](notes/project.md), and the weekly roadmap is in [`notes/plan.md`](notes/plan.md). All write-ups are listed in the [notes index](notes/README.md).

## Start here

1. **Set up** (local): `conda env create -f environment.yml`, then copy `.env.example` to `.env` and fill in `ROBOFLOW_EXPORT_URL`. On Colab, add the same name under Secrets. The first cell of each notebook installs the package.
2. **Run the notebooks in order:**

| Notebook | What it does | Notes |
|---|---|---|
| [`00_download_data.ipynb`](notebooks/00_download_data.ipynb) | Downloads the 500 Roboflow images into `data/` | [`notes/00_data.md`](notes/00_data.md) |
| [`01_eda_overview.ipynb`](notebooks/01_eda_overview.ipynb) | Exploratory data analysis: counts, Live/Dead, image quality, overlap, colour | |
| [`02_baseline_cellpose.ipynb`](notebooks/02_baseline_cellpose.ipynb) | Baseline: off-the-shelf Cellpose-SAM counts cells (GPU recommended) | [`02_baseline_results`](notes/02_baseline_results.md), [`02_baseline_explained`](notes/02_baseline_explained.md), [`02_baseline_slides`](notes/02_baseline_slides.md) |
| [`03_initial_model_train.ipynb`](notebooks/03_initial_model_train.ipynb) | First YOLO26n model trained on our images | |
| [`03a_rgb_grayscale.ipynb`](notebooks/03a_rgb_grayscale.ipynb) | YOLO26n trained on RGB vs grayscale input | [`03a_rgb_grayscale_results`](notes/03a_rgb_grayscale_results.md), [`03a_rgb_grayscale_diagnostics`](notes/03a_rgb_grayscale_diagnostics.md) |
| [`03b_augmentation_exploration.ipynb`](notebooks/03b_augmentation_exploration.ipynb) | Effect of training augmentations (colour, flips) | |
| [`04_model_application.ipynb`](notebooks/04_model_application.ipynb) | Applies the trained model to the synthetic cell dataset | |
| [`05_extra_experimentation.ipynb`](notebooks/05_extra_experimentation.ipynb) | Extra detection tests on other datasets (blood cells, tomatoes, eggs) | |

## Layout

```
notebooks/        notebooks that run the analysis
src/cellcount/    reusable code (data loading, EDA, splits, metrics, models)
splits/           splits.csv: the shared train/valid/test split, which every model must use
notes/            write-ups numbered like their notebooks; start at notes/README.md
data/, outputs/   gitignored: downloaded images, results, model caches
```
