"""
Shared helpers for the evaluation scripts (evaluate_seeds.py, lstm_sweep.py).

Kept separate from compare_models.py on purpose: that script is the published,
single-seed benchmark and should stay byte-for-byte reproducible. These helpers
generalize its split logic to any seed and add the truncated-point metrics.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from save_model import cmapss_score  # noqa: E402

DATA_PATH = os.path.join(ROOT, "nasa_cmapss_FD001_scaled.csv")
ID_COL = "unit_number"
CYCLE_COL = "time_in_cycles"
DOCS = os.path.join(ROOT, "docs")


def split_engines(df, seed, val_fraction=0.2):
    """Engine-grouped train/test split plus a validation carve-out of train.

    With seed=42 this reproduces compare_models.split_engines() exactly, so a
    seed sweep that includes 42 lines up with the published numbers.

    Returns (train_engines, fit_engines, val_engines, test_engines).
    """
    groups = df[ID_COL].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_idx, te_idx = next(gss.split(df, groups=groups))

    train_engines = np.unique(groups[tr_idx])
    test_engines = np.unique(groups[te_idx])

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_engines)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_engines = np.sort(shuffled[:n_val])
    fit_engines = np.sort(shuffled[n_val:])
    return train_engines, fit_engines, val_engines, test_engines


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "CMAPSS": float(cmapss_score(y_true, y_pred)),
    }


def truncated_metrics(units, y_true, y_pred, seed, draws=200):
    """One-prediction-per-engine score, the way the official benchmark works.

    The official C-MAPSS test set cuts every engine off at some point and scores
    the model ONLY at that last visible cycle. We only have full run-to-failure
    trajectories, so we imitate it: for each draw, pick one random evaluation
    point per test engine, score those points, and average over many draws.

    Returns
      trunc_RMSE   mean RMSE over the draws (cycles)
      trunc_score  mean asymmetric C-MAPSS score PER ENGINE (the official score
                   is a sum over engines, so this is that score divided by the
                   engine count, which makes it comparable across split sizes)

    This approximates the official protocol; it is not identical to it, so do
    not compare the numbers directly to published leaderboard values.
    """
    rng = np.random.default_rng(seed)
    unique_units = np.unique(units)
    index_by_unit = [np.flatnonzero(units == u) for u in unique_units]

    rmses, scores = [], []
    for _ in range(draws):
        pick = np.array([rng.choice(idx) for idx in index_by_unit])
        yt, yp = y_true[pick], y_pred[pick]
        rmses.append(np.sqrt(np.mean((yp - yt) ** 2)))
        scores.append(cmapss_score(yt, yp) / len(pick))
    return {"trunc_RMSE": float(np.mean(rmses)),
            "trunc_score": float(np.mean(scores))}
