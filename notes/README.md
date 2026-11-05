# Notes

The team's write-ups. Each note starts with the number of the notebook it documents, so `02_baseline_results.md` goes with `notebooks/02_baseline_cellpose.ipynb`. Each note also opens with a header line that gives its notebook and links back here.

## Overview

- [Project](project.md): scope, data sources, stakeholders, KPIs
- [Plan](plan.md): weekly roadmap, one row per check-in

## Contents, in notebook order

| Notebook | What it does | Notes |
|---|---|---|
| [`00_download_data`](../notebooks/00_download_data.ipynb) | Downloads the 500 Roboflow images; the shared split | [`00_data.md`](00_data.md): dataset, download, train/valid/test split |
| [`01_eda_overview`](../notebooks/01_eda_overview.ipynb) | Exploratory data analysis | No note yet |
| [`02_baseline_cellpose`](../notebooks/02_baseline_cellpose.ipynb) | Baseline: off-the-shelf Cellpose-SAM | [`02_baseline_results.md`](02_baseline_results.md): B0–B3 scores and figures<br>[`02_baseline_explained.md`](02_baseline_explained.md): plain-language guide to each model and metric<br>[`02_baseline_slides.md`](02_baseline_slides.md): two-slide summary with script and Q&A |
| [`03_initial_model_train`](../notebooks/03_initial_model_train.ipynb) | First YOLO26n model trained on our images | No note yet |
| [`03a_rgb_grayscale`](../notebooks/03a_rgb_grayscale.ipynb) | YOLO26n on RGB vs grayscale input | [`03a_rgb_grayscale_results.md`](03a_rgb_grayscale_results.md): does grayscale change count accuracy?<br>[`03a_rgb_grayscale_diagnostics.md`](03a_rgb_grayscale_diagnostics.md): per-source errors and example images |
| [`03b_augmentation_exploration`](../notebooks/03b_augmentation_exploration.ipynb) | Effect of training augmentations | No note yet |
| [`04_model_application`](../notebooks/04_model_application.ipynb) | Applies the trained model to the synthetic cell dataset | No note yet |
| [`05_extra_experimentation`](../notebooks/05_extra_experimentation.ipynb) | Extra detection tests on other datasets | No note yet |

## Folders

```
figures/
  baseline/         figures in 02_baseline_results
  slides/           figures in 02_baseline_slides, plus make_slide_figs.py, which regenerates them
  rgb_grayscale/    figures in 03a_rgb_grayscale_*
results/
  rgb_grayscale/    CSV and JSON results behind 03a_rgb_grayscale_*
```

## Adding a note

Name it `<notebook number>_<topic>.md`, copy the header line from an existing note, and fill in its row above.
