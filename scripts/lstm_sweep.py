"""
LSTM architecture sweep: window length, hidden size, depth, dropout.

Round 5 of the model-selection story showed that more epochs do not help and
named the next honest step: change the architecture or window. This script does
that, one factor at a time around the deployed configuration
(window 30, hidden 64, 2 layers, dropout 0.2), so each change is attributable.

Protocol
  * One fixed engine split (seed 42, same as compare_models.py).
  * Configs are CHOSEN by validation RMSE (early-stopping engines), never by the
    test engines, so the test numbers are not tuned on.
  * Every config is scored on the SAME test points (cycles >= the largest window
    in the grid), so a longer window cannot win by skipping hard early cycles.
  * Each config is a single training run. Seed noise is roughly +/-0.35 RMSE
    (see scripts/evaluate_seeds.py), so gaps smaller than that are not evidence.

Usage
  python scripts/lstm_sweep.py                 # full one-factor-at-a-time grid
  python scripts/lstm_sweep.py --max-epochs 80 --patience 12   # faster, rougher

Output: docs/lstm_sweep.csv
"""

import argparse
import os
import time

import _eval_utils as eu
import numpy as np
import pandas as pd

from lstm_model import (DROPOUT, EPOCHS, HIDDEN_SIZE, NUM_LAYERS, PATIENCE,
                        SEQ_LEN, make_windows, predict, train_lstm)

BASE = {"seq_len": SEQ_LEN, "hidden": HIDDEN_SIZE, "layers": NUM_LAYERS,
        "dropout": DROPOUT}

CONFIGS = [
    {},                              # the deployed configuration
    {"seq_len": 20}, {"seq_len": 40}, {"seq_len": 50},
    {"hidden": 32}, {"hidden": 128},
    {"layers": 1}, {"layers": 3},
    {"dropout": 0.0}, {"dropout": 0.4},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(eu.DATA_PATH)
    _, fit_e, val_e, test_e = eu.split_engines(df, args.seed)
    configs = [{**BASE, **c} for c in CONFIGS]
    shared_min_cycle = max(c["seq_len"] for c in configs)

    rows = []
    for i, cfg in enumerate(configs, 1):
        label = (f"win{cfg['seq_len']}_h{cfg['hidden']}_l{cfg['layers']}"
                 f"_d{cfg['dropout']}")
        print(f"[{i}/{len(configs)}] {label}", flush=True)

        Xf, yf, _, _ = make_windows(df, fit_e, seq_len=cfg["seq_len"])
        Xv, yv, _, _ = make_windows(df, val_e, seq_len=cfg["seq_len"])
        Xt, yt, _, _ = make_windows(df, test_e, seq_len=cfg["seq_len"],
                                    min_cycle=shared_min_cycle)

        t0 = time.time()
        model, hist = train_lstm(
            Xf, yf, Xv, yv, epochs=args.max_epochs, patience=args.patience,
            seed=args.seed, verbose=False, hidden_size=cfg["hidden"],
            num_layers=cfg["layers"], dropout=cfg["dropout"])
        secs = time.time() - t0

        h = pd.DataFrame(hist)
        best = h.sort_values("val_rmse").iloc[0]
        m = eu.regression_metrics(yt, predict(model, Xt))
        rows.append({
            "config": label, **cfg, "best_epoch": int(best["epoch"]),
            "val_RMSE": float(best["val_rmse"]), "test_RMSE": m["RMSE"],
            "test_MAE": m["MAE"], "test_R2": m["R2"],
            "test_CMAPSS": m["CMAPSS"], "seconds": round(secs),
        })
        print(f"    val {best['val_rmse']:.2f}  test RMSE {m['RMSE']:.2f}  "
              f"({secs:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(eu.DOCS, "lstm_sweep.csv"),
                                  index=False)

    res = pd.DataFrame(rows)
    chosen = res.loc[res["val_RMSE"].idxmin()]
    base = res.iloc[0]
    print("\n" + "=" * 72)
    print(f"All configs scored on identical test points "
          f"(cycles >= {shared_min_cycle}, {len(yt):,} points)")
    print("=" * 72)
    print(res[["config", "best_epoch", "val_RMSE", "test_RMSE", "test_MAE",
               "test_R2"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nBest by VALIDATION RMSE: {chosen['config']} "
          f"(test RMSE {chosen['test_RMSE']:.2f} vs deployed config "
          f"{base['test_RMSE']:.2f}, difference {base['test_RMSE'] - chosen['test_RMSE']:+.2f})")
    print("Single run per config: differences under ~0.35 RMSE are within seed noise.")
    print("\nSaved docs/lstm_sweep.csv")


if __name__ == "__main__":
    main()
