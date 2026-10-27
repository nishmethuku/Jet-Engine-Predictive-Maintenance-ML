"""
LSTM RUL regressor for NASA C-MAPSS FD001.

Why an LSTM at all: the linear/logistic/RF models see ONE cycle at a time —
a single row of 16 scaled sensor readings — and have to guess remaining life
from that snapshot. Degradation is a trajectory, so an LSTM that reads the
last SEQ_LEN cycles of an engine can use the *trend* (how fast sensor 11 is
drifting), not just the current value.

Windowing: for engine u and cycle t, the sample is rows [t-SEQ_LEN+1 .. t]
and the label is that engine's capped RUL at cycle t. Engines shorter than
SEQ_LEN cycles are left-padded with their first row (FD001's shortest engine
is 128 cycles, so this is just a safety net).

Trained/evaluated by compare_models.py. Nothing here reads or writes the
dashboard's model file — app.py is untouched.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import FEATURE_COLS, RUL_CAP

SEQ_LEN = 30        # cycles of history fed to the LSTM per prediction
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Round 5 schedule. The first run capped at 60 epochs and stopped with
# validation RMSE still falling — early stopping never fired, so 16.59 was a
# floor rather than the model's ceiling. This schedule runs long enough for the
# curve to actually flatten, and decays the learning rate on plateau so the
# tail of training can refine instead of bouncing around the minimum.
EPOCHS = 300
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PATIENCE = 25       # early-stopping patience, in epochs
LR_FACTOR = 0.5     # multiply LR by this when validation RMSE plateaus
LR_PATIENCE = 8     # epochs of no improvement before decaying
MIN_LR = 1e-5       # stop decaying here
SEED = 42

ID_COL = "unit_number"
CYCLE_COL = "time_in_cycles"


def make_windows(df, engine_ids, seq_len=SEQ_LEN, min_cycle=None):
    """Build (X, y, units, cycles) sliding windows for the given engines.

    Returns
      X      float32 [n, seq_len, n_features]
      y      float32 [n]                       capped RUL at the window's last cycle
      units  int     [n]                       engine id of each window
      cycles int     [n]                       last cycle of each window

    `min_cycle` drops windows whose end cycle is below it. compare_models.py
    passes min_cycle=seq_len so every model is scored on prediction points
    backed by a full, unpadded history window.
    """
    Xs, ys, us, cs = [], [], [], []

    for unit in engine_ids:
        eng = df[df[ID_COL] == unit].sort_values(CYCLE_COL)
        feats = eng[FEATURE_COLS].to_numpy(dtype=np.float32)
        rul = np.minimum(eng["RUL"].to_numpy(dtype=np.float32), RUL_CAP)
        cycles = eng[CYCLE_COL].to_numpy()

        for i in range(len(eng)):
            if min_cycle is not None and cycles[i] < min_cycle:
                continue
            start = i - seq_len + 1
            if start < 0:
                # left-pad by repeating the engine's first reading
                pad = np.repeat(feats[:1], -start, axis=0)
                window = np.concatenate([pad, feats[: i + 1]], axis=0)
            else:
                window = feats[start : i + 1]
            Xs.append(window)
            ys.append(rul[i])
            us.append(unit)
            cs.append(cycles[i])

    return (
        np.stack(Xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(us),
        np.asarray(cs),
    )


class LSTMRegressor(nn.Module):
    """2-layer LSTM -> dropout -> linear head, predicting a single RUL value.

    Only the final timestep's hidden state feeds the head: that step has seen
    the whole window, so it carries the summarized degradation trend.
    """

    def __init__(self, n_features, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def train_lstm(X_tr, y_tr, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE,
               lr=LEARNING_RATE, patience=PATIENCE, seed=SEED, verbose=True,
               hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
    """Train with MSE + Adam, early-stopping on validation RMSE.

    The architecture arguments default to the module constants, so existing
    callers behave exactly as before; scripts/lstm_sweep.py overrides them.

    Returns (model, history):
      model    best validation weights restored, so a lucky final epoch can't
               inflate the reported test numbers
      history  list of {"epoch", "train_mse", "val_rmse"} — the learning curve
               the README plots, logged rather than transcribed by hand
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMRegressor(n_features=X_tr.shape[2], hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val)

    best_rmse = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            # LSTMs on long-ish windows can spike gradients; clip for stability
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_pred = torch.clamp(model(Xv), 0, RUL_CAP)
            val_rmse = torch.sqrt(loss_fn(val_pred, yv)).item()

        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_mse": total / len(y_tr),
            "val_rmse": val_rmse,
            "lr": current_lr,
        })

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  train MSE {total / len(y_tr):7.2f}"
                  f"   val RMSE {val_rmse:6.2f}   lr {current_lr:.1e}")

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best val RMSE {best_rmse:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict(model, X, batch_size=1024):
    """Batched inference, clipped to the valid RUL range."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            chunk = torch.from_numpy(X[i : i + batch_size])
            out.append(model(chunk).numpy())
    return np.clip(np.concatenate(out), 0, RUL_CAP)
