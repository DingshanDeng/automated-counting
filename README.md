# fall-2026-automated-cell-counting
Team project: fall-2026-automated-cell-counting

A deep-learning tool that counts cells in microscopy images. Scope and KPIs are in [`notes/project.md`](notes/project.md), and the weekly roadmap is in [`notes/plan.md`](notes/plan.md).

## Start here

1. **Set up** (local): `conda env create -f environment.yml`, then copy `.env.example` to `.env` and fill in `ROBOFLOW_EXPORT_URL`. On Colab, add the same name under Secrets. The first cell of each notebook installs the package.
2. **Run the notebooks in order:**

| Notebook | What it does | Write-up |
|---|---|---|
| [`00_download_data.ipynb`](notebooks/00_download_data.ipynb) | Downloads the 500 Roboflow images into `data/` | [`notes/data.md`](notes/data.md) |
| [`01_eda_overview.ipynb`](notebooks/01_eda_overview.ipynb) | Exploratory data analysis: counts, Live/Dead, image quality, overlap, colour | [`notes/data.md`](notes/data.md) |
| [`02_baseline_cellpose.ipynb`](notebooks/02_baseline_cellpose.ipynb) | Baseline: off-the-shelf Cellpose-SAM counts cells (GPU recommended) | [`notes/baseline_results.md`](notes/baseline_results.md), [`notes/baseline_explained.md`](notes/baseline_explained.md) |

## Layout

```
notebooks/        notebooks that run the analysis
src/cellcount/    reusable code (data loading, EDA, splits, metrics, models)
splits/           splits.csv: the shared train/valid/test split, which every model must use
notes/            plan, project scope, and write-ups with figures
data/, outputs/   gitignored: downloaded images, results, model caches
```
