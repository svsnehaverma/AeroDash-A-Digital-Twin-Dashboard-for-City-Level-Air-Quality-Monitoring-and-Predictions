"""
Forecast all sensors into ONE CSV (raw timestamps, fast & robust).

- Input:  /mnt/data/PM1 - PM1_till_19_Aug.csv   (columns: Timestamp, Sensor Name, Value)
- Output: /mnt/data/PM1_all_sensors_72h_forecasts_raw.csv

Key features:
  • NO resampling or averaging; uses your native sampling.
  • Converts 72 hours into steps based on your median sampling interval.
  • Adds last HISTORY_HOURS of history in the output.
  • Models: HW (when freq is regular), Linear, XGBoost (if installed), LSTM (if torch installed).
  • Faster: trains on last TRAIN_TAIL_POINTS points only (trim, not downsample).
  • Suppresses HW convergence warnings; graceful fallbacks to simpler ETS when needed.
"""

import os, re, time, warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# ---- Optional deps flags ----
_HAVE_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    _HAVE_XGB = False

_HAVE_TORCH = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    _HAVE_TORCH = False

_HAVE_HW = True
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
except Exception:
    _HAVE_HW = False

from pandas.tseries.frequencies import to_offset

# ---------- Config ----------
DATA_PATH = "PM1 - PM1_till_19_Aug.csv"
OUT_FILE  = "PM1_all_sensors_72h_forecasts_raw.csv"

STEPS_HOURS       = 72        # forecast horizon in HOURS
HISTORY_HOURS     = 24 * 7    # last 7 days of history in output
MIN_POINTS_SENSOR = 100       # skip very short series

# Speed settings (no downsampling; just trimming tail)
TRAIN_TAIL_POINTS = 3000      # use only last N points for model fitting (set None to use all)
EPOCHS_LSTM       = 6         # was 12
XGB_TREES         = 100       # was 150

# Enable/disable models quickly
ENABLE_HW   = True
ENABLE_LR   = True
ENABLE_XGB  = True
ENABLE_LSTM = True

# ---------- Utils ----------
def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def load_raw_series(df: pd.DataFrame, sensor: str) -> pd.Series:
    """Return raw timestamp series (sorted), no resampling."""
    sdf = df[df["Sensor Name"] == sensor][["Timestamp", "Value"]].dropna().copy()
    if sdf.empty:
        return pd.Series(dtype=float)
    sdf["Timestamp"] = pd.to_datetime(sdf["Timestamp"])
    ts = sdf.set_index("Timestamp")["Value"].sort_index()
    return ts

def sampling_info(ts: pd.Series) -> Tuple[Optional[pd.Timedelta], Optional[float], Optional[str]]:
    """Estimate median sampling interval, samples per hour, and pandas freq string (if inferable)."""
    if ts.index.size < 2:
        return None, None, None
    # infer frequency string (works only if regular)
    freq = pd.infer_freq(ts.index)
    # median delta in seconds
    deltas = np.diff(ts.index.values).astype("timedelta64[s]").astype(np.int64)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None, None, freq
    med_sec = float(np.median(deltas))
    delta = pd.Timedelta(seconds=med_sec)
    samples_per_hour = 3600.0 / med_sec
    return delta, samples_per_hour, freq

def steps_for_horizon(ts: pd.Series, hours: int) -> int:
    delta, sph, _ = sampling_info(ts)
    if delta is None or sph is None or sph <= 0:
        return max(1, hours)  # fallback: 1 step/hour
    return int(round(hours * sph))

def seasonal_periods_for_hw(freq: str) -> Optional[int]:
    """Approximate daily seasonality in native steps; needs a valid pandas freq string."""
    try:
        off = to_offset(freq)
        # steps per day = 24h / step
        per_day = int(round(pd.Timedelta(days=1) / off.delta))
        return per_day if per_day >= 2 else None
    except Exception:
        return None

def history_slice(ts: pd.Series, hours: int) -> pd.Series:
    if ts.empty:
        return ts
    last_time = ts.index.max()
    return ts[ts.index >= (last_time - pd.Timedelta(hours=hours))]

def tail_for_training(ts: pd.Series) -> pd.Series:
    if TRAIN_TAIL_POINTS is None:
        return ts
    return ts.tail(TRAIN_TAIL_POINTS)

# ---------- Models ----------
def forecast_hw(ts: pd.Series, steps: int) -> np.ndarray:
    if not (_HAVE_HW and ENABLE_HW):
        return np.full(steps, np.nan)

    # HW needs a regular freq; skip if not inferable
    _, _, freq = sampling_info(ts)
    if not freq:
        return np.full(steps, np.nan)

    sp = seasonal_periods_for_hw(freq)
    ts_reg = ts.asfreq(freq)
    if sp is None or sp >= max(3, ts_reg.size):
        # try simpler models if seasonal period is not meaningful
        try:
            fit = ExponentialSmoothing(ts_reg, trend="add", seasonal=None).fit(optimized=True, use_brute=True)
            return np.asarray(fit.forecast(steps).values, dtype=float)
        except Exception:
            try:
                fit = ExponentialSmoothing(ts_reg, trend=None, seasonal=None).fit(optimized=True, use_brute=True)
                return np.asarray(fit.forecast(steps).values, dtype=float)
            except Exception:
                return np.full(steps, np.nan)

    # Full seasonal model
    try:
        fit = ExponentialSmoothing(ts_reg, trend="add", seasonal="add", seasonal_periods=sp).fit(
            optimized=True, use_brute=True
        )
        return np.asarray(fit.forecast(steps).values, dtype=float)
    except Exception:
        # fallbacks
        try:
            fit = ExponentialSmoothing(ts_reg, trend="add", seasonal=None).fit(optimized=True, use_brute=True)
            return np.asarray(fit.forecast(steps).values, dtype=float)
        except Exception:
            try:
                fit = ExponentialSmoothing(ts_reg, trend=None, seasonal=None).fit(optimized=True, use_brute=True)
                return np.asarray(fit.forecast(steps).values, dtype=float)
            except Exception:
                return np.full(steps, np.nan)

def forecast_linear(ts: pd.Series, steps: int) -> np.ndarray:
    if not ENABLE_LR:
        return np.full(steps, np.nan)
    tr = tail_for_training(ts).values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(tr.reshape(-1,1)).ravel()
    X = np.arange(len(y)).reshape(-1,1)
    model = LinearRegression().fit(X, y)
    X_future = np.arange(len(y), len(y)+steps).reshape(-1,1)
    y_pred_scaled = model.predict(X_future)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    return y_pred

def forecast_xgb(ts: pd.Series, steps: int) -> np.ndarray:
    if not (_HAVE_XGB and ENABLE_XGB):
        return np.full(steps, np.nan)
    tr = tail_for_training(ts).values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(tr.reshape(-1,1)).ravel()
    X = np.arange(len(y)).reshape(-1,1)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=XGB_TREES,
        max_depth=4,
        n_jobs=-1
    )
    model.fit(X, y, verbose=False)
    X_future = np.arange(len(y), len(y)+steps).reshape(-1,1)
    y_pred_scaled = model.predict(X_future)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    return y_pred

if _HAVE_TORCH and ENABLE_LSTM:
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)

    class TimeSeriesDataset(Dataset):
        def __init__(self, series, seq_len=24):
            self.series = series.astype(np.float32)
            self.seq_len = seq_len
        def __len__(self):
            return len(self.series) - self.seq_len
        def __getitem__(self, idx):
            x = self.series[idx:idx+self.seq_len]
            y = self.series[idx+self.seq_len]
            return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

def forecast_lstm(ts: pd.Series, steps: int, seq_len: int = 24) -> np.ndarray:
    if not (_HAVE_TORCH and ENABLE_LSTM):
        return np.full(steps, np.nan)

    tr = tail_for_training(ts).values
    if len(tr) <= seq_len + 1:
        return np.full(steps, np.nan)

    scaler = MinMaxScaler()
    y = scaler.fit_transform(tr.reshape(-1,1)).ravel()

    dataset = TimeSeriesDataset(y, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = LSTMPredictor()
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(EPOCHS_LSTM):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    preds_scaled = []
    context = y[-seq_len:].tolist()
    with torch.no_grad():
        for _ in range(steps):
            seq = torch.tensor(context[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            nxt = model(seq).item()
            preds_scaled.append(nxt)
            context.append(nxt)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).ravel()
    return preds

# ---------- Run ----------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if not {"Timestamp", "Sensor Name", "Value"}.issubset(df.columns):
        raise ValueError("Expected columns: 'Timestamp', 'Sensor Name', 'Value'")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    sensors = sorted(df["Sensor Name"].dropna().unique().tolist())
    all_rows = []

    print(f"Found {len(sensors)} sensors. TRAIN_TAIL_POINTS={TRAIN_TAIL_POINTS}, LSTM_EPOCHS={EPOCHS_LSTM}, XGB_TREES={XGB_TREES}")
    for sensor in sensors:
        t0 = time.time()
        ts = load_raw_series(df, sensor)
        if ts.size < MIN_POINTS_SENSOR:
            print(f"[skip] {sensor}: only {ts.size} points (<{MIN_POINTS_SENSOR})")
            continue

        steps = steps_for_horizon(ts, STEPS_HOURS)
        if steps <= 0:
            print(f"[skip] {sensor}: steps={steps}")
            continue

        hist = history_slice(ts, HISTORY_HOURS)

        # Forecasts
        hw   = forecast_hw(ts, steps)
        lr   = forecast_linear(ts, steps)
        xgb  = forecast_xgb(ts, steps)
        lstm = forecast_lstm(ts, steps)

        # Future index based on median native delta
        delta, _, _ = sampling_info(ts)
        if delta is None:
            delta = pd.Timedelta(hours=1)
        last_time = ts.index.max()
        future_index = [last_time + (i+1)*delta for i in range(steps)]

        # Assemble output
        hist_df = pd.DataFrame({
            "Timestamp": hist.index,
            "Sensor Name": sensor,
            "Type": "historical",
            "Actual": hist.values,
            "HW": np.nan,
            "LR": np.nan,
            "XGB": np.nan,
            "LSTM": np.nan
        })
        fc_df = pd.DataFrame({
            "Timestamp": future_index,
            "Sensor Name": sensor,
            "Type": "forecast",
            "Actual": np.nan,
            "HW": hw,
            "LR": lr,
            "XGB": xgb,
            "LSTM": lstm
        })

        all_rows.append(hist_df)
        all_rows.append(fc_df)
        dt = time.time() - t0
        print(f"[done] {sensor}: {len(ts)} pts -> {steps} steps in {dt:.1f}s")

    if not all_rows:
        raise RuntimeError("No sensor produced output (too few points or errors).")

    result = pd.concat(all_rows, ignore_index=True)
    result["Timestamp"] = pd.to_datetime(result["Timestamp"])
    result.sort_values(["Sensor Name", "Timestamp", "Type"], inplace=True)
    result.to_csv(OUT_FILE, index=False)
    print(f"\nSaved ALL sensors to: {OUT_FILE}")

if __name__ == "__main__":
    main()

