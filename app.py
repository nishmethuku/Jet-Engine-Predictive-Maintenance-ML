"""
RUL Prediction Dashboard — Team 04C
AI4ALL Ignite SU26 | NASA C-MAPSS Turbofan

Combines the engine-browser view with CSV-upload and manual-input
prediction into one app, all backed by the same model (see model.py).

Run with: streamlit run app.py
Requires: streamlit, numpy, pandas, scikit-learn, joblib
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

from model import (
    FEATURE_COLS, LSTM_SEQ_LEN, RUL_CAP,
    load_lstm, load_model, looks_unscaled, normalize_columns, predict_rul,
    predict_rul_sequence, predict_rul_sequences,
)

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"

st.set_page_config(page_title="Turbofan RUL Predictor", page_icon="✈️", layout="wide")


@st.cache_resource
def get_model():
    # Prefer a pre-trained pickle (fast). On a fresh deployment the .pkl isn't
    # in the repo (it's gitignored and >100MB), so train it once from the
    # committed CSV — cached for the life of the process. Import sklearn lazily
    # and tolerate failure so an environment issue degrades to the placeholder
    # heuristic instead of crashing the whole app.
    try:
        model = load_model()
        if model is None and os.path.exists(DATA_PATH):
            from save_model import train_full_model
            with st.spinner("First run: training the model (~10s)…"):
                model = train_full_model(DATA_PATH)
        return model
    except Exception as exc:  # noqa: BLE001 - surface, don't crash
        st.warning(f"Model unavailable ({type(exc).__name__}); using placeholder.")
        return None


@st.cache_resource
def get_lstm():
    """The LSTM is the primary model, but it's optional at runtime: it needs
    torch and a checkpoint. If either is missing the app serves Random Forest
    predictions instead of failing."""
    return load_lstm()


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    data = pd.read_csv(DATA_PATH)
    # Standardize the id columns so the rest of the app can use unit/cycle.
    data = data.rename(columns={"unit_number": "unit", "time_in_cycles": "cycle"})
    # The CSV ships raw RUL; apply the same 125-cycle cap the model trains on.
    data["RUL"] = data["RUL"].clip(upper=RUL_CAP)
    return data


@st.cache_data
def engine_life_predictions(engine_id, _lstm):
    """LSTM prediction at every cycle of one engine (cached per engine).

    `_lstm` has a leading underscore so Streamlit does not try to hash the
    model object; the cache key is just the engine id.
    """
    life = dataset[dataset["unit"] == engine_id].sort_values("cycle")
    return life["cycle"].to_numpy(), predict_rul_sequences(life, _lstm)


def example_csv():
    """Two engines' first 60 cycles, in the exact format the uploader expects."""
    if dataset is None:
        return None
    sample = dataset[dataset["unit"].isin([1, 2]) & (dataset["cycle"] <= 60)]
    return sample[["unit", "cycle"] + FEATURE_COLS].to_csv(index=False).encode("utf-8")


model = get_model()
lstm = get_lstm()
dataset = load_dataset()

with st.sidebar:
    st.title("✈️ Turbofan RUL Predictor")
    st.caption("Team 04C · AI4ALL Ignite SU26 · NASA C-MAPSS FD001")
    st.markdown(
        "Predicting a turbofan engine's Remaining Useful Life (RUL) from "
        "sensor telemetry, to support predictive maintenance instead of "
        "fixed replacement schedules."
    )
    if model is None:
        st.warning(
            f"No model and no `{DATA_PATH}` to train from — showing "
            "placeholder predictions."
        )
    st.divider()
    if lstm is not None:
        st.success(
            f"**LSTM active** — used whenever {LSTM_SEQ_LEN}+ cycles of history "
            "are available. Random Forest covers the rest."
        )
    else:
        st.info(
            "**Random Forest active.** The LSTM is unavailable here (needs "
            "`torch` and `lstm_rul_model.pt`)."
        )
    mode = st.radio("Mode", ["Browse engine", "Upload CSV", "Manual input"])

# ── Browse engine ────────────────────────────────────────────────────────────
if mode == "Browse engine":
    st.header("Browse a training-set engine")
    if dataset is None:
        st.error(f"`{DATA_PATH}` not found in the repo root.")
    else:
        engine_ids = sorted(dataset["unit"].unique().tolist())
        selected_engine = st.selectbox("Select engine", engine_ids)

        engine_df = dataset[dataset["unit"] == selected_engine].sort_values("cycle")
        min_cycle = int(engine_df["cycle"].min())
        max_cycle = int(engine_df["cycle"].max())
        st.caption(f"{len(engine_df)} recorded cycles ({min_cycle}-{max_cycle}).")

        selected_cycle = st.slider(
            "Cycle", min_value=min_cycle, max_value=max_cycle, value=max_cycle
        )
        current_row = engine_df.loc[engine_df["cycle"] == selected_cycle].iloc[[0]]

        # The LSTM reads the run-up to this cycle, not just the cycle itself —
        # that trajectory is exactly what makes it more accurate than the RF.
        history = engine_df[engine_df["cycle"] <= selected_cycle]
        lstm_rul = predict_rul_sequence(history, lstm)

        if lstm_rul is not None:
            predicted_rul, used = lstm_rul, "LSTM"
        else:
            predicted_rul, used = predict_rul(current_row, model)[0], "Random Forest"

        col1, col2, col3 = st.columns(3)
        col1.metric("Current cycle", selected_cycle)
        col2.metric("Predicted RUL (cycles)", f"{predicted_rul:.0f}")
        col3.metric("Model used", used)

        if used == "LSTM":
            n = min(len(history), LSTM_SEQ_LEN)
            st.caption(
                f"Prediction made from the last **{n} cycles** of this engine's "
                f"history"
                + ("" if n == LSTM_SEQ_LEN else
                   f" (padded to {LSTM_SEQ_LEN} — the engine is younger than the "
                   f"window)")
                + "."
            )

        if lstm is not None:
            st.subheader("Predicted vs true RUL over this engine's life")
            cycles, life_pred = engine_life_predictions(selected_engine, lstm)
            life_chart = pd.DataFrame({
                "True RUL (capped at 125)": engine_df["RUL"].to_numpy(),
                "LSTM prediction": life_pred,
            }, index=pd.Index(cycles, name="cycle"))
            st.line_chart(life_chart)
            st.caption(
                "The deployed LSTM was trained on all 100 engines, including this "
                "one, so the fit here looks better than it would on a brand-new "
                "engine. Held-out accuracy is in the footer and the README."
            )

        st.subheader("Sensor readings")
        sensor_cols = [c for c in FEATURE_COLS if c.startswith("sensor_")]
        chosen_sensors = st.multiselect(
            "Sensors to plot", sensor_cols, default=sensor_cols[:4]
        )
        if chosen_sensors:
            st.line_chart(engine_df.set_index("cycle")[chosen_sensors])
        st.caption("Sensor values are z-score scaled, not raw physical units.")

# ── Upload CSV ───────────────────────────────────────────────────────────────
elif mode == "Upload CSV":
    st.header("Upload sensor data for live predictions")
    st.caption(
        "CSV must be pre-scaled (z-score standardized) with columns: "
        + ", ".join(FEATURE_COLS)
        + ". Add `unit` and `cycle` columns (`unit_number` and `time_in_cycles` "
        "also work) to get LSTM predictions."
    )
    sample = example_csv()
    if sample is not None:
        st.download_button(
            "Download an example CSV", sample, "example_engines.csv", "text/csv"
        )
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded:
        upload_df = normalize_columns(pd.read_csv(uploaded))
        st.write("Preview:", upload_df.head())

        missing = [c for c in FEATURE_COLS if c not in upload_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            feats = upload_df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
            bad_rows = int(feats.isna().any(axis=1).sum())
            if bad_rows:
                st.error(
                    f"{bad_rows} row(s) have blank or non-numeric values in the "
                    "feature columns. Fix or remove them and upload again."
                )
                st.stop()
            upload_df[FEATURE_COLS] = feats

            unscaled = looks_unscaled(upload_df)
            if unscaled:
                st.warning(
                    "These columns look like raw sensor units, not z-scores: "
                    + ", ".join(unscaled)
                    + ". The models expect standardized inputs, so predictions "
                    "for this file will not be meaningful."
                )

            # The LSTM needs to know which rows belong to which engine and in
            # what order. With unit+cycle columns we can rebuild each row's
            # history and use it; without them, every row is an isolated
            # snapshot and only the Random Forest can score it.
            has_sequence = {"unit", "cycle"}.issubset(upload_df.columns)

            lstm_preds = None
            if lstm is not None and has_sequence:
                with st.spinner("Running the LSTM over each engine's history…"):
                    lstm_preds = predict_rul_sequences(upload_df, lstm)

            if lstm_preds is not None:
                upload_df["Predicted RUL"] = np.round(lstm_preds, 1)
                upload_df["Model"] = "LSTM"
                upload_df = upload_df.sort_values(["unit", "cycle"]).reset_index(drop=True)
                short = int(
                    (upload_df.groupby("unit").cumcount() + 1 < LSTM_SEQ_LEN).sum()
                )
                st.success(
                    f"Predictions complete — **LSTM** used for all "
                    f"{len(upload_df):,} rows, reading each engine's history."
                )
                if short:
                    st.info(
                        f"{short:,} row(s) come from an engine's first "
                        f"{LSTM_SEQ_LEN - 1} cycles, so their window was padded "
                        "with the engine's earliest reading. Treat those "
                        "predictions as rough."
                    )
            else:
                upload_df["Predicted RUL"] = predict_rul(upload_df, model)
                upload_df["Model"] = "Random Forest"
                if lstm is not None and not has_sequence:
                    st.info(
                        "Used the **Random Forest**: this CSV has no `unit` and "
                        "`cycle` columns, so there's no way to tell which rows "
                        "form an engine's history. Add those two columns to get "
                        "LSTM predictions."
                    )
                else:
                    st.success("Predictions complete!")

            id_cols = [c for c in ("unit", "cycle") if c in upload_df.columns]
            show = id_cols + ["Predicted RUL", "Model"]
            st.dataframe(upload_df[show] if id_cols else upload_df[["Predicted RUL", "Model"]])
            csv_out = upload_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV", csv_out, "rul_predictions.csv", "text/csv"
            )

# ── Manual input ─────────────────────────────────────────────────────────────
else:
    st.header("Manual sensor input")
    st.caption("Enter a single engine's sensor readings (z-score standardized).")

    if lstm is not None:
        st.info(
            f"**This mode uses the Random Forest, not the LSTM** — and that's a "
            f"limitation, not a bug. The LSTM predicts from a **trajectory**: it "
            f"needs the last {LSTM_SEQ_LEN} cycles to see how fast the sensors are "
            f"drifting. A single hand-entered snapshot has no history, so only a "
            f"model that reads one cycle at a time can score it. Use **Browse "
            f"engine** or upload a CSV with `unit`/`cycle` columns for LSTM "
            f"predictions."
        )

    col1, col2 = st.columns(2)
    inputs = {}
    for i, feat in enumerate(FEATURE_COLS):
        col = col1 if i % 2 == 0 else col2
        inputs[feat] = col.number_input(feat, value=0.0, format="%.4f")

    if st.button("Predict RUL", type="primary"):
        row = pd.DataFrame([inputs])
        unscaled = looks_unscaled(row, threshold=10.0)
        if unscaled:
            st.warning(
                "These values look like raw sensor units, not z-scores: "
                + ", ".join(unscaled)
                + ". The model expects standardized inputs, so this prediction "
                "will not be meaningful."
            )
        pred = predict_rul(row, model)[0]
        st.metric("Predicted Remaining Useful Life", f"{pred:.1f} cycles")
        if pred < 30:
            st.error("⚠️ Engine approaching end of life — maintenance recommended soon.")
        elif pred < 60:
            st.warning("\U0001f536 Moderate degradation detected — schedule inspection.")
        else:
            st.success("✅ Engine health looks good.")

st.divider()
if lstm is not None:
    st.caption(
        "Primary model: **LSTM** over a 30-cycle window — RMSE 16.19 ± 0.35, "
        "MAE 11.59 ± 0.53, R² 0.850 on unseen engines (mean of 3 environments; "
        "see docs/reproduction-log.md). Across 5 different engine splits its "
        "RMSE is 16.1 ± 1.1 and it beat the Random Forest (about 19.0) on every "
        "split. Random Forest serves rows with no usable history. RUL capped at "
        "125 cycles."
    )
else:
    st.caption(
        "Model: tuned Random Forest (RMSE 18.69, MAE 13.82, R² 0.80 on a grouped "
        "FD001 hold-out). RUL capped at 125 cycles."
    )
