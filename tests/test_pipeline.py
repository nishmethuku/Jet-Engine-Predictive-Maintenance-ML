"""
Regression tests for the RUL pipeline. Run with:  python -m pytest -q

They cover the places this project can silently go wrong: window construction
(off-by-one or padding errors would not crash, they would just hurt accuracy),
agreement between the per-row and batched LSTM predictors, the scoring
function's asymmetry, the checkpoint, and the dataset's label invariants.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

torch = pytest.importorskip("torch")

from lstm_model import LSTMRegressor, SEQ_LEN, make_windows  # noqa: E402
from model import (  # noqa: E402
    FEATURE_COLS, LSTM_MODEL_PATH, RUL_CAP, load_lstm, looks_unscaled,
    normalize_columns, predict_rul_sequence, predict_rul_sequences,
)
from save_model import cmapss_score  # noqa: E402

DATA = os.path.join(ROOT, "nasa_cmapss_FD001_scaled.csv")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(0)
    m = LSTMRegressor(n_features=len(FEATURE_COLS))
    m.eval()
    return m


# ── dataset invariants ───────────────────────────────────────────────────────
def test_labels_are_consistent(df):
    """RUL must equal (last cycle - current cycle) for every engine."""
    last = df.groupby("unit_number")["time_in_cycles"].transform("max")
    assert (df["RUL"] == last - df["time_in_cycles"]).all()


def test_no_missing_values(df):
    assert not df[FEATURE_COLS + ["RUL"]].isna().any().any()


# ── windowing ────────────────────────────────────────────────────────────────
def test_window_shapes_and_label_cap(df):
    X, y, units, cycles = make_windows(df, [1, 2])
    assert X.shape == (len(y), SEQ_LEN, len(FEATURE_COLS))
    assert len(units) == len(cycles) == len(y)
    assert y.max() <= RUL_CAP


def test_window_ends_on_its_own_cycle(df):
    """The last row of each window must be that sample's own cycle."""
    X, _, units, cycles = make_windows(df, [1], min_cycle=SEQ_LEN)
    eng = df[df["unit_number"] == 1].set_index("time_in_cycles")
    for i in (0, 10, len(cycles) - 1):
        np.testing.assert_allclose(
            X[i, -1], eng.loc[cycles[i], FEATURE_COLS].to_numpy(dtype=np.float32),
            rtol=1e-6)


def test_young_engine_is_left_padded_with_first_reading(df):
    X, _, _, cycles = make_windows(df, [1])
    first = df[(df["unit_number"] == 1) & (df["time_in_cycles"] == 1)]
    first = first[FEATURE_COLS].to_numpy(dtype=np.float32)[0]
    assert cycles[0] == 1
    np.testing.assert_allclose(X[0, 0], first, rtol=1e-6)   # padded
    np.testing.assert_allclose(X[0, -1], first, rtol=1e-6)  # the real reading


# ── batched predictor must match the per-row predictor ───────────────────────
def test_batched_matches_per_row(df, tiny_model):
    sub = df[df["unit_number"].isin([3, 7])].rename(
        columns={"unit_number": "unit", "time_in_cycles": "cycle"})
    batched = predict_rul_sequences(sub, tiny_model)
    for unit, grp in sub.groupby("unit", sort=False):
        grp = grp.sort_values("cycle")
        for i in (0, 5, 29, 30, len(grp) - 1):
            one = predict_rul_sequence(grp.iloc[: i + 1], tiny_model)
            pos = grp.index[i]
            assert batched[sub.index.get_loc(pos)] == pytest.approx(one, abs=1e-4)


def test_batched_respects_original_row_order(df, tiny_model):
    sub = df[df["unit_number"].isin([3, 7])].rename(
        columns={"unit_number": "unit", "time_in_cycles": "cycle"})
    shuffled = sub.sample(frac=1.0, random_state=0)
    a = pd.Series(predict_rul_sequences(sub, tiny_model), index=sub.index)
    b = pd.Series(predict_rul_sequences(shuffled, tiny_model), index=shuffled.index)
    pd.testing.assert_series_equal(a, b.loc[a.index], check_exact=False, atol=1e-4)


def test_batched_returns_none_without_a_model(df):
    assert predict_rul_sequences(df.head(5), None) is None


# ── deployed checkpoint ──────────────────────────────────────────────────────
def test_deployed_checkpoint_loads_and_predicts_in_range(df):
    assert os.path.exists(os.path.join(ROOT, LSTM_MODEL_PATH))
    old = os.getcwd()
    os.chdir(ROOT)  # model.py resolves the checkpoint relative to the cwd
    try:
        lstm = load_lstm()
    finally:
        os.chdir(old)
    assert lstm is not None
    sub = df[df["unit_number"] == 1].rename(
        columns={"unit_number": "unit", "time_in_cycles": "cycle"})
    preds = predict_rul_sequences(sub, lstm)
    assert preds.shape == (len(sub),)
    assert preds.min() >= 0 and preds.max() <= RUL_CAP
    # An engine near the end of life must be predicted as close to failure.
    assert preds[-1] < 30


# ── scoring ──────────────────────────────────────────────────────────────────
def test_cmapss_score_penalizes_late_more_than_early():
    y = np.array([50.0])
    late = cmapss_score(y, y + 15)    # predicted to last LONGER (dangerous)
    early = cmapss_score(y, y - 15)
    assert late > early > 0
    assert cmapss_score(y, y) == 0


# ── input helpers ────────────────────────────────────────────────────────────
def test_normalize_columns_accepts_both_spellings():
    a = normalize_columns(pd.DataFrame(columns=["unit_number", "time_in_cycles"]))
    b = normalize_columns(pd.DataFrame(columns=["unit", "cycle"]))
    assert list(a.columns) == list(b.columns) == ["unit", "cycle"]


def test_looks_unscaled_flags_raw_units_only(df):
    assert looks_unscaled(df.head(200)) == []
    raw = df.head(5).copy()
    raw["sensor_2"] = 642.5
    assert looks_unscaled(raw) == ["sensor_2"]
