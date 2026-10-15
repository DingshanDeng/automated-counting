"""Counting and detection metrics (plan.md §4).

Counting metrics take a per-image frame with ``total_true, total_pred`` (``total_count_metrics``)
or ``live_true, dead_true, live_pred, dead_pred`` (``count_metrics``, the Live/Dead extension).
Detection metrics take box frames in the ``load_coco`` schema (``uid, category, x, y, w, h``),
with an optional ``conf`` column on the predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("Live", "Dead")


def per_image_counts(boxes: pd.DataFrame, uids: pd.Series | list[str]) -> pd.DataFrame:
    """Live and Dead counts per image, with 0 for images that have no boxes of a class."""
    counts = boxes.groupby(["uid", "category"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=pd.Index(uids, name="uid"), columns=list(CLASSES), fill_value=0)
    return counts.rename(columns={"Live": "live", "Dead": "dead"}).reset_index()


def _viability(live: pd.Series, dead: pd.Series) -> pd.Series:
    total = live + dead
    return 100 * live / total.where(total > 0)


def total_count_metrics(df: pd.DataFrame) -> pd.Series:
    """Total-count MAE, bias (mean signed error), RMSE, relative error, ±10% accuracy and R²."""
    err = df["total_pred"] - df["total_true"]
    ss_tot = ((df["total_true"] - df["total_true"].mean()) ** 2).sum()
    return pd.Series(
        {
            "n_images": len(df),
            "mae_total": err.abs().mean(),
            "bias_total": err.mean(),
            "rmse_total": np.sqrt((err**2).mean()),
            "rel_err_total": (err.abs() / df["total_true"].where(df["total_true"] > 0)).mean(),
            "within_10pct": (err.abs() <= 0.1 * df["total_true"]).mean(),
            "r2_total": 1 - (err**2).sum() / ss_tot if ss_tot > 0 else np.nan,
        }
    )


def count_metrics(df: pd.DataFrame) -> pd.Series:
    """MAE (Live, Dead, Total), viability error, ±10% accuracy, R², RMSE and relative error."""
    total_true = df["live_true"] + df["dead_true"]
    total_pred = df["live_pred"] + df["dead_pred"]
    err = total_pred - total_true
    ss_tot = ((total_true - total_true.mean()) ** 2).sum()
    viab_err = (_viability(df["live_pred"], df["dead_pred"]) - _viability(df["live_true"], df["dead_true"])).abs()
    return pd.Series(
        {
            "n_images": len(df),
            "mae_live": (df["live_pred"] - df["live_true"]).abs().mean(),
            "mae_dead": (df["dead_pred"] - df["dead_true"]).abs().mean(),
            "mae_total": err.abs().mean(),
            "bias_total": err.mean(),
            "rmse_total": np.sqrt((err**2).mean()),
            "rel_err_total": (err.abs() / total_true.where(total_true > 0)).mean(),
            "within_10pct": (err.abs() <= 0.1 * total_true).mean(),
            "r2_total": 1 - (err**2).sum() / ss_tot if ss_tot > 0 else np.nan,
            "viability_err_pp": viab_err.mean(),
        }
    )


def count_metrics_by(df: pd.DataFrame, by: str | pd.Series, metrics=count_metrics) -> pd.DataFrame:
    """``metrics`` (default ``count_metrics``) per group, e.g. per ``source`` or per count bin."""
    return df.groupby(by, observed=True).apply(metrics, include_groups=False)


def _cross_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` and every box in ``b`` (rows of ``x, y, w, h``)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def match_boxes(pred: pd.DataFrame, gt: pd.DataFrame, iou: float = 0.5, by_class: bool = True) -> pd.DataFrame:
    """Greedy one-to-one matching, most confident prediction first.

    Returns one row per image (and class, unless ``by_class=False``) with ``tp, fp, fn``.
    """
    keys = ["uid", "category"] if by_class else ["uid"]
    if "conf" not in pred:
        pred = pred.assign(conf=1.0)
    pred_groups = dict(list(pred.groupby(keys)))
    gt_groups = dict(list(gt.groupby(keys)))
    rows = []
    for key in sorted(set(pred_groups) | set(gt_groups)):
        p = pred_groups.get(key, pred.iloc[:0]).sort_values("conf", ascending=False)
        g = gt_groups.get(key, gt.iloc[:0])
        tp = 0
        if len(p) and len(g):
            ious = _cross_iou(p[["x", "y", "w", "h"]].to_numpy(float), g[["x", "y", "w", "h"]].to_numpy(float))
            taken = np.zeros(len(g), dtype=bool)
            for row in ious:
                row = np.where(taken, -1.0, row)
                j = row.argmax()
                if row[j] >= iou:
                    taken[j] = True
                    tp += 1
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), "tp": tp, "fp": len(p) - tp, "fn": len(g) - tp})
    return pd.DataFrame(rows)


def detection_metrics(pred: pd.DataFrame, gt: pd.DataFrame, iou: float = 0.5) -> pd.DataFrame:
    """Precision, recall and F1 per class, plus a class-agnostic ``Cell`` row."""
    per_class = match_boxes(pred, gt, iou).groupby("category")[["tp", "fp", "fn"]].sum()
    agnostic = match_boxes(pred, gt, iou, by_class=False)[["tp", "fp", "fn"]].sum().rename("Cell")
    out = pd.concat([per_class.reindex(list(CLASSES), fill_value=0), agnostic.to_frame().T])
    out["precision"] = out["tp"] / (out["tp"] + out["fp"]).replace(0, np.nan)
    out["recall"] = out["tp"] / (out["tp"] + out["fn"]).replace(0, np.nan)
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    return out.rename_axis("class")
