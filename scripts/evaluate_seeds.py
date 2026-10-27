"""
Multi-seed evaluation: is the LSTM's lead over the baselines real, or split luck?

The published comparison (compare_models.py) uses ONE engine split, seed 42.
With only 25 test engines, a different split can move RMSE by a couple of
cycles, so a single seed cannot tell a real margin from luck. This script
repeats the whole experiment over several seeds. Each seed changes BOTH the
engine split and the LSTM's initialization, then reports mean +/- std.

For every seed it trains Linear regression, the Random Forest and the LSTM on
the same engines and scores them on the same test points (cycles >= SEQ_LEN),
the fairness rule compare_models.py already uses. It reports two views:

  all points   every shared test cycle (the project's existing metric)
  truncated    ONE random cycle per test engine, averaged over many draws, which
               imitates how the official C-MAPSS benchmark scores a model

Usage
  python scripts/evaluate_seeds.py                     # seeds 42..46
  python scripts/evaluate_seeds.py --seeds 42 7 13     # chosen seeds
  python scripts/evaluate_seeds.py --max-epochs 100    # cap LSTM training time

Output: docs/seed_evaluation.csv (per seed), docs/seed_evaluation_summary.csv
"""

import argparse
import os
import time

import _eval_utils as eu
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lstm_model import EPOCHS, SEQ_LEN, make_windows, predict, train_lstm
from model import FEATURE_COLS, RUL_CAP
from save_model import build_model


def eval_points(df, engines):
    """Shared test points (cycle >= SEQ_LEN): flat rows, LSTM windows, targets."""
    Xw, y, units, cycles = make_windows(df, engines, min_cycle=SEQ_LEN)
    indexed = df.set_index([eu.ID_COL, eu.CYCLE_COL]).sort_index()
    rows = indexed.loc[list(zip(units, cycles))]
    return rows[FEATURE_COLS].to_numpy(), Xw, y, units


def flat_train(df, engines):
    sub = df[df[eu.ID_COL].isin(engines)]
    return sub[FEATURE_COLS].to_numpy(), np.minimum(sub["RUL"].to_numpy(), RUL_CAP)


def run_seed(df, seed, max_epochs):
    train_e, fit_e, val_e, test_e = eu.split_engines(df, seed)
    X_tr, y_tr = flat_train(df, train_e)
    X_te, Xw_te, y_te, units_te = eval_points(df, test_e)

    preds = {}
    preds["LINEAR"] = np.clip(LinearRegression().fit(X_tr, y_tr).predict(X_te),
                              0, RUL_CAP)
    preds["RF"] = np.clip(build_model().fit(X_tr, y_tr).predict(X_te), 0, RUL_CAP)

    Xw_fit, y_fit, _, _ = make_windows(df, fit_e)
    Xw_val, y_val, _, _ = make_windows(df, val_e)
    t0 = time.time()
    lstm, history = train_lstm(Xw_fit, y_fit, Xw_val, y_val, epochs=max_epochs,
                               seed=seed, verbose=False)
    preds["LSTM"] = predict(lstm, Xw_te)
    best_epoch = int(pd.DataFrame(history).sort_values("val_rmse").iloc[0]["epoch"])
    print(f"  LSTM trained in {time.time() - t0:.0f}s "
          f"({len(history)} epochs, best epoch {best_epoch})", flush=True)

    rows = []
    for name, p in preds.items():
        row = {"seed": seed, "model": name, **eu.regression_metrics(y_te, p),
               **eu.truncated_metrics(units_te, y_te, p, seed)}
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--max-epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    df = pd.read_csv(eu.DATA_PATH)
    all_rows = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        all_rows += run_seed(df, seed, args.max_epochs)
        pd.DataFrame(all_rows).to_csv(
            os.path.join(eu.DOCS, "seed_evaluation.csv"), index=False)

    res = pd.DataFrame(all_rows)
    metrics = ["RMSE", "MAE", "R2", "CMAPSS", "trunc_RMSE", "trunc_score"]
    summary = res.groupby("model")[metrics].agg(["mean", "std"])
    summary = summary.loc[["LINEAR", "RF", "LSTM"]]
    flat = pd.DataFrame(index=summary.index)
    for m in metrics:
        flat[m + "_mean"] = summary[(m, "mean")]
        flat[m + "_std"] = summary[(m, "std")]
    flat.to_csv(os.path.join(eu.DOCS, "seed_evaluation_summary.csv"))

    print("\n" + "=" * 72)
    print(f"MEAN +/- STD OVER {len(args.seeds)} SEEDS (seeds {args.seeds})")
    print("=" * 72)
    for name in ["LINEAR", "RF", "LSTM"]:
        parts = [f"{m} {flat.loc[name, m + '_mean']:.2f}"
                 f" +/- {flat.loc[name, m + '_std']:.2f}" for m in
                 ("RMSE", "MAE", "trunc_RMSE")]
        print(f"{name:7s} " + "   ".join(parts) +
              f"   R2 {flat.loc[name, 'R2_mean']:.3f}")

    piv = res.pivot(index="seed", columns="model", values="RMSE")
    diff = piv["RF"] - piv["LSTM"]
    print(f"\nLSTM beats RF on RMSE in {(diff > 0).sum()}/{len(diff)} seeds "
          f"(mean margin {diff.mean():+.2f} cycles, worst {diff.min():+.2f})")
    piv_l = piv["LINEAR"] - piv["LSTM"]
    print(f"LSTM beats LINEAR on RMSE in {(piv_l > 0).sum()}/{len(piv_l)} seeds "
          f"(mean margin {piv_l.mean():+.2f} cycles, worst {piv_l.min():+.2f})")
    print("\nSaved docs/seed_evaluation.csv, docs/seed_evaluation_summary.csv")


if __name__ == "__main__":
    main()
