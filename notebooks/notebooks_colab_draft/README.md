# Colab drafts

Colab versions of the local notebooks, plus a training notebook. Upload this whole folder to Google Drive and open the notebooks from there.

| Notebook | What it does | GPU |
|---|---|---|
| `00_setup_and_download.ipynb` | Downloads the Roboflow export into `data/` on Drive | no |
| `02_baselines.ipynb` | Reruns baselines B0, B1, B4 (Cellpose-SAM) and B7 (YOLO26n) | yes |
| `03_yolo_finetune.ipynb` | Trains and compares YOLO experiments (model size, augmentation), with resume after a disconnect and optional leave-one-session-out | yes |

## Upload

1. Locally, refresh the code copies. Rerun this after any change to `src/` or `splits/`:
   ```bash
   notebooks/notebooks_colab_draft/sync_from_local.sh
   ```
2. In Google Drive, create `My Drive/cell-image-counting/` and upload the **contents** of this folder into it:
   ```
   My Drive/cell-image-counting/
   ├── pyproject.toml
   ├── src/cellcount/*.py
   ├── splits/splits.csv
   ├── 00_setup_and_download.ipynb
   ├── 02_baselines.ipynb
   └── 03_yolo_finetune.ipynb
   ```
   To use another folder, change `DRIVE_ROOT` in each notebook's first cell.
3. Optional, to skip work:
   - Upload `data/roboflow-cell-counting-raw-coco/` to `cell-image-counting/data/`, and you can skip notebook 00.
   - Upload `outputs/` to `cell-image-counting/outputs/` to reuse the cached Cellpose masks and the trained `yolo26n_default` weights.
4. In Colab, add the `ROBOFLOW_EXPORT_URL` secret (key icon in the left sidebar, with notebook access on). Choose **Runtime > Change runtime type > T4 GPU** for notebooks 02 and 03.
5. Open a notebook from Drive (double-click, then "Open with Google Colaboratory") and click **Run all**.

## How it works on Colab

- **Bootstrap (first cell):** mounts Drive, copies `src/` and `pyproject.toml` to `/content`, and runs `pip install -e ".[model]"`.
- **Storage:** sets `CELLCOUNT_STORAGE` to the Drive folder, so `data/`, `outputs/` and `splits/` all resolve there.
- **Images:** copied to `/content/work` at the start of each session. Drive is slow for many small files and doesn't support the symlinks that `export_yolo` makes.
- **Weights, results and figures:** written to `outputs/` on Drive, so they survive a disconnect. Rerunning notebook 03 skips finished runs and resumes interrupted ones from `last.pt`.

After you edit code locally, rerun the sync script, upload the changed files again, then restart the Colab runtime.

The copies in `src/`, `splits/` and `pyproject.toml` are gitignored like the originals, so only the notebooks, this README and the sync script are tracked here.
