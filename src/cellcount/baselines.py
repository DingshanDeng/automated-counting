"""Reference baselines that predict counts without looking at the cells (models B0, B1).

Both take per-image frames with ``live_true, dead_true`` (and ``source`` for B1) and return the
test frame with ``live_pred, dead_pred`` added.
"""

from __future__ import annotations

import pandas as pd


def predict_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """B0: every test image gets the training-set mean Live and Dead counts."""
    return test.assign(live_pred=train["live_true"].mean(), dead_pred=train["dead_true"].mean())


def predict_session_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """B1: every test image gets its session's training mean (the global mean for unseen sessions).

    On the random split this measures the "recognize the session" shortcut; under
    leave-one-session-out it falls back to B0.
    """
    means = train.groupby("source")[["live_true", "dead_true"]].mean()
    live = test["source"].map(means["live_true"]).fillna(train["live_true"].mean())
    dead = test["source"].map(means["dead_true"]).fillna(train["dead_true"].mean())
    return test.assign(live_pred=live, dead_pred=dead)
