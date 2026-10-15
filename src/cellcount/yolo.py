"""YOLO Live/Dead detector (plan.md B7): export our split to YOLO format, train, predict, tune.

Ultralytics is imported inside the functions, so ``cellcount`` stays importable without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import CLASSES

SPLIT_NAMES = ("train", "valid", "test")


def export_yolo(images: pd.DataFrame, boxes: pd.DataFrame, splits: pd.DataFrame, out: Path) -> Path:
    """Write ``out/{train,valid,test}/{images,labels}`` and ``out/data.yaml`` following ``splits``.

    Images are symlinked, not copied. Labels are ``class cx cy w h``, normalized to [0, 1], with
    classes in ``metrics.CLASSES`` order (0 = Live, 1 = Dead).
    """
    out = Path(out)
    images = images.drop(columns="split").merge(splits[["uid", "split"]], on="uid")
    class_id = {c: i for i, c in enumerate(CLASSES)}
    by_uid = dict(list(boxes.groupby("uid")))
    for row in images.itertuples():
        img_dir, lbl_dir = out / row.split / "images", out / row.split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = row.uid.replace("/", "_")
        link = img_dir / f"{stem}{Path(row.path).suffix}"
        if not link.exists():
            link.symlink_to(Path(row.path).resolve())
        b = by_uid.get(row.uid, boxes.iloc[:0])
        lines = [
            f"{class_id[c]} {cx / row.width:.6f} {cy / row.height:.6f} {w / row.width:.6f} {h / row.height:.6f}"
            for c, cx, cy, w, h in zip(b["category"], b["cx"], b["cy"], b["w"], b["h"])
        ]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n{names}\n")
    return yaml


def train(data_yaml: Path, project: Path, model: str = "yolo26n.pt", name: str = "baseline", **kwargs) -> Path:
    """Fine-tune a COCO-pretrained YOLO with Ultralytics defaults; returns the best weights."""
    from ultralytics import YOLO

    args = {"epochs": 100, "imgsz": 640, "patience": 20, "device": _device(), "seed": 0, "plots": True}
    args.update(kwargs)
    result = YOLO(model).train(data=str(data_yaml), project=str(project), name=name, exist_ok=True, **args)
    return Path(result.save_dir) / "weights" / "best.pt"


def predict(weights: Path, images: pd.DataFrame, conf: float = 0.05, batch: int = 16) -> pd.DataFrame:
    """Boxes in the ``load_coco`` schema plus ``conf``, for every image (low ``conf`` keeps
    candidates so the threshold can be tuned afterwards with ``tune_conf``)."""
    from ultralytics import YOLO

    net = YOLO(str(weights))
    rows = []
    uids, paths = images["uid"].tolist(), images["path"].tolist()
    for i in range(0, len(paths), batch):
        results = net.predict(paths[i:i + batch], conf=conf, max_det=1000, device=_device(), verbose=False)
        for uid, r in zip(uids[i:i + batch], results):
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2), c, k in zip(xyxy, r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                rows.append({"uid": uid, "category": CLASSES[k], "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "conf": c})
    pred = pd.DataFrame(rows, columns=["uid", "category", "x", "y", "w", "h", "conf"])
    pred["cx"], pred["cy"] = pred["x"] + pred["w"] / 2, pred["y"] + pred["h"] / 2
    return pred


def tune_conf(pred: pd.DataFrame, truth: pd.DataFrame, grid: np.ndarray | None = None) -> float:
    """Confidence threshold minimizing total-count MAE. ``truth`` has ``uid, live_true, dead_true``;
    use validation images only."""
    from .metrics import per_image_counts

    grid = np.round(np.arange(0.05, 0.91, 0.01), 2) if grid is None else grid
    total_true = (truth["live_true"] + truth["dead_true"]).to_numpy()
    mae = []
    for t in grid:
        c = per_image_counts(pred[pred["conf"] >= t], truth["uid"])
        mae.append(np.abs(c["live"] + c["dead"] - total_true).mean())
    return float(grid[int(np.argmin(mae))])


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "0" if torch.cuda.is_available() else "cpu"
