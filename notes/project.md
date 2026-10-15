# Cell Image Counting

## Scope

Counting cells in microscopy images is still mostly done by hand, which is slow and tedious. We are building a deep-learning tool that takes a microscopy image and outputs the cell count. It finds each cell, then counts what it found, so every count can be checked against the image. We want it to work on images from different cell sources and imaging conditions. We test it explicitly on the hard cases: high cell density, overlapping cells, and poor image quality such as blur.

## Data sources

| Dataset | Status | Description |
|---|---|---|
| [Roboflow Cell Counting](https://universe.roboflow.com/cell-counting/cell-counting-zeqo5) | **In use** | 500 brightfield images (640×640) from 5 acquisition sessions, with 23–91 cells each, boxed as Live or Dead. See [`data.md`](data.md). |
| Synthetic Cell Images and Masks | Planned | 20,400 synthetic images at known blur levels, labelled with cell counts |
| CoNIC | Planned | Histology nuclei: dense, crowded scenes |

## Stakeholders

- **Pathology labs and medical companies:** automated counts for diagnostics and quality control.
- **Research labs:** cell-culture monitoring and experiments that depend on counts.

## KPIs

| KPI | Measure | Baseline now (Cellpose-SAM, 75 test images) |
|---|---|---|
| Count accuracy | Mean absolute error in cells per image; share of images within ±10% | 5.6 cells; 59% |
| R² | Variance in true counts that the predictions explain | 0.74 |
| Compute | Seconds per image, compared with counting by hand (about 1 minute) | 1.85 s |

Details: [`baseline_results.md`](baseline_results.md). Roadmap: [`plan.md`](plan.md).
