#!/usr/bin/env python3
"""
Predictions with native sampling (no resampling), 2 days validation + 2 days future.

Input  : PM1 - PM1_till_19_Aug.csv
         Required columns: Timestamp, Sensor Name, Value
         Optional columns: Sensor Centroid Latitude, Sensor Centroid Longitude
Output : PM1_all_sensors_2d_val_2d_future_raw.csv

What it does:
  • Keeps original timestamps (no resampling / averaging).
  • Validation: hold out last 48h (based on your native sampling), predict it, compute MSE & MAPE.
  • Future: train on the full series and predict the next 48h beyond the last timestamp.
  • Models: Holt–Winters (only if regular freq can be inferred), Linear Regression, XGBoost (if installed), LSTM (if torch installed).
  • Adds (Latitude, Longitude, Formatted Address) if lat/lon columns exist. Geocoding is rate-limited and optional.

Note:
  • HW needs a regular frequency. If pandas cannot infer a freq (irregular timestamps), HW is skipped (NaNs in HW column).
  • Future forecasts are trained on the WHOLE dataset (no tail trimming).
"""

import os, re, time, warnings
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---- Optional dependencies ----
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

_HAVE_GEOPY = True
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.exc import GeocoderServiceError
except Exception:
    _HAVE_GEOPY = False

from pandas.tseries.frequencies import to_offset


# ============================== Config ==============================
DATA_PATH = "PM1 - PM1_till_19_Aug.csv"
OUT_FILE  = "PM1_all_sensors_2d_val_2d_future_raw_NEW.csv"

HISTORY_HOURS     = 24 * 7     # how much history to include in output (for plotting context)
VALIDATION_HOURS  = 48         # 2 days validation window
FUTURE_HOURS      = 48         # 2 days future forecast

MIN_POINTS_SENSOR = 100        # skip very short series

# Train on WHOLE data for future forecasts (per your request)
TRAIN_TAIL_POINTS = None       # None => use ALL points

# LSTM settings (kept modest)
EPOCHS_LSTM       = 8
LSTM_SEQ_LEN      = 96         # context window length (in native steps)
BATCH_SIZE        = 32
LR_LSTM           = 1e-3

# XGBoost settings
XGB_TREES         = 120
XGB_MAX_DEPTH     = 4

# Quick toggles
ENABLE_HW   = True
ENABLE_LR   = True
ENABLE_XGB  = True
ENABLE_LSTM = True
# ===================================================================


# ============================== Utils ==============================
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def safe_mape(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def load_raw_series(df: pd.DataFrame, sensor: str) -> pd.Series:
    """Return a value series with original timestamps, sorted; no resampling."""
    sdf = df[df["Sensor Name"] == sensor][["Timestamp", "Value"]].dropna().copy()
    if sdf.empty:
        return pd.Series(dtype=float)
    sdf["Timestamp"] = pd.to_datetime(sdf["Timestamp"])
    return sdf.set_index("Timestamp")["Value"].sort_index()

def sampling_info(ts: pd.Series) -> Tuple[Optional[pd.Timedelta], Optional[float], Optional[str]]:
    """Estimate median sampling interval, samples/hour, and inferable pandas freq string."""
    if ts.index.size < 2:
        return None, None, None
    # Try to infer a regular pandas frequency
    freq = pd.infer_freq(ts.index)
    # Median delta (robust to occasional gaps)
    deltas = np.diff(ts.index.values).astype("timedelta64[s]").astype(np.int64)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None, None, freq
    med_sec = float(np.median(deltas))
    delta = pd.Timedelta(seconds=med_sec)
    sph = 3600.0 / med_sec
    return delta, sph, freq

def steps_for_hours(ts: pd.Series, hours: int) -> int:
    delta, sph, _ = sampling_info(ts)
    if delta is None or sph is None or sph <= 0:
        return max(1, hours)  # fallback: 1 step/hour
    return int(round(hours * sph))

def history_slice(ts: pd.Series, hours: int) -> pd.Series:
    if ts.empty:
        return ts
    last_time = ts.index.max()
    return ts[ts.index >= (last_time - pd.Timedelta(hours=hours))]

def tail_for_training(ts: pd.Series) -> pd.Series:
    if TRAIN_TAIL_POINTS is None:
        return ts
    return ts.tail(TRAIN_TAIL_POINTS)
# ===================================================================


# ============================ Geocoding ============================
if _HAVE_GEOPY:
    geolocator = Nominatim(user_agent="ncl_pm1_forecast_native")
    reverse_geo = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)
else:
    geolocator = None
    reverse_geo = None

def format_address(location) -> Optional[str]:
    if not location:
        return None
    address = location.raw.get('address', {})
    road = address.get('road', '')
    neighbourhood = address.get('neighbourhood', '') or address.get('suburb', '')
    city = address.get('city', '') or address.get('town', '') or address.get('village', '')
    county = address.get('county', '')
    state = address.get('state', '')
    country = address.get('country', '')
    postcode = address.get('postcode', '')
    parts = [road, neighbourhood, city, county, state]
    if postcode:
        parts.append(postcode)
    if country:
        parts.append(country)
    return ', '.join([p for p in parts if p])

def safe_reverse_geocode(lat, lon) -> Optional[str]:
    if not _HAVE_GEOPY or pd.isna(lat) or pd.isna(lon):
        return None
    try:
        loc = reverse_geo((lat, lon), language="en")
        return format_address(loc) if loc else None
    except GeocoderServiceError:
        return None
    except Exception:
        return None
# ===================================================================


# ============================== Models =============================
from pandas.tseries.frequencies import to_offset

def seasonal_periods_for_hw(freq: str) -> Optional[int]:
    try:
        off = to_offset(freq)
        per_day = int(round(pd.Timedelta(days=1) / off.delta))
        return per_day if per_day >= 2 else None
    except Exception:
        return None

def fcst_hw(train: pd.Series, steps: int) -> np.ndarray:
    if not (_HAVE_HW and ENABLE_HW) or train.size < 10:
        return np.full(steps, np.nan)
    # HW requires a regular index; if it can't be inferred, skip
    _, _, freq = sampling_info(train)
    if not freq:
        return np.full(steps, np.nan)
    sp = seasonal_periods_for_hw(freq)
    tr_reg = train.asfreq(freq)
    # Try seasonal; then trend-only; then simple
    try:
        if sp is not None and sp < max(3, tr_reg.size):
            fit = ExponentialSmoothing(tr_reg, trend="add", seasonal="add", seasonal_periods=sp).fit(
                optimized=True, use_brute=True
            )
            return np.asarray(fit.forecast(steps).values, dtype=float)
    except Exception:
        pass
    try:
        fit = ExponentialSmoothing(tr_reg, trend="add", seasonal=None).fit(optimized=True, use_brute=True)
        return np.asarray(fit.forecast(steps).values, dtype=float)
    except Exception:
        pass
    try:
        fit = ExponentialSmoothing(tr_reg, trend=None, seasonal=None).fit(optimized=True, use_brute=True)
        return np.asarray(fit.forecast(steps).values, dtype=float)
    except Exception:
        return np.full(steps, np.nan)

def fcst_lr(train: pd.Series, steps: int) -> np.ndarray:
    if not ENABLE_LR:
        return np.full(steps, np.nan)
    tr = tail_for_training(train).values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(tr.reshape(-1,1)).ravel()
    X = np.arange(len(y)).reshape(-1,1)
    model = LinearRegression().fit(X, y)
    X_future = np.arange(len(y), len(y)+steps).reshape(-1,1)
    y_pred_scaled = model.predict(X_future)
    return scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

def fcst_xgb(train: pd.Series, steps: int) -> np.ndarray:
    if not (_HAVE_XGB and ENABLE_XGB):
        return np.full(steps, np.nan)
    tr = tail_for_training(train).values
    scaler = MinMaxScaler()
    y = scaler.fit_transform(tr.reshape(-1,1)).ravel()
    X = np.arange(len(y)).reshape(-1,1)
    model = XGBRegressor(objective="reg:squarederror", n_estimators=XGB_TREES, max_depth=XGB_MAX_DEPTH, n_jobs=-1)
    model.fit(X, y, verbose=False)
    X_future = np.arange(len(y), len(y)+steps).reshape(-1,1)
    y_pred_scaled = model.predict(X_future)
    return scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

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
        def __init__(self, series, seq_len=LSTM_SEQ_LEN):
            self.series = series.astype(np.float32)
            self.seq_len = seq_len
        def __len__(self):
            return len(self.series) - self.seq_len
        def __getitem__(self, idx):
            x = self.series[idx:idx+self.seq_len]
            y = self.series[idx+self.seq_len]
            return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

def fcst_lstm(train: pd.Series, steps: int, seq_len: int = LSTM_SEQ_LEN) -> np.ndarray:
    if not (_HAVE_TORCH and ENABLE_LSTM):
        return np.full(steps, np.nan)
    tr_vals = tail_for_training(train).values
    if len(tr_vals) <= seq_len + 1:
        return np.full(steps, np.nan)

    scaler = MinMaxScaler()
    y = scaler.fit_transform(tr_vals.reshape(-1,1)).ravel()

    dataset = TimeSeriesDataset(y, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LSTMPredictor()
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR_LSTM)

    model.train()
    for _ in range(EPOCHS_LSTM):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

    # Recursive forecast
    model.eval()
    preds_scaled = []
    context = y[-seq_len:].tolist()
    with torch.no_grad():
        for _ in range(steps):
            seq = torch.tensor(context[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            nxt = model(seq).item()
            preds_scaled.append(nxt)
            context.append(nxt)

    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).ravel()
# ===================================================================


# =============================== Run ===============================
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    need = {"Timestamp", "Sensor Name", "Value"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns: {need - set(df.columns)}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    has_latlon = {"Sensor Centroid Latitude", "Sensor Centroid Longitude"}.issubset(df.columns)

    sensors = sorted(df["Sensor Name"].dropna().unique().tolist())
    all_rows = []

    print(f"Found {len(sensors)} sensors. Training future forecasts on the whole dataset.")
    for sensor in sensors:
        t0 = time.time()
        ts = load_raw_series(df, sensor)
        if ts.size < MIN_POINTS_SENSOR:
            print(f"[skip] {sensor}: {ts.size} points (<{MIN_POINTS_SENSOR})")
            continue

        steps_val    = steps_for_hours(ts, VALIDATION_HOURS)
        steps_future = steps_for_hours(ts, FUTURE_HOURS)
        if steps_val <= 0 or steps_future <= 0 or ts.size <= steps_val + 10:
            print(f"[skip] {sensor}: not enough data for {VALIDATION_HOURS}h validation")
            continue

        # Geocoding once per sensor
        lat = df.loc[df["Sensor Name"]==sensor, "Sensor Centroid Latitude"].iloc[0] if has_latlon else None
        lon = df.loc[df["Sensor Name"]==sensor, "Sensor Centroid Longitude"].iloc[0] if has_latlon else None
        formatted_addr = safe_reverse_geocode(lat, lon)

        # ----- Validation (backtest) -----
        train_bt = ts.iloc[:-steps_val]
        test_bt  = ts.iloc[-steps_val:]
        preds_val: Dict[str, np.ndarray] = {
            "HW":   fcst_hw(train_bt, steps_val),
            "LR":   fcst_lr(train_bt, steps_val),
            "XGB":  fcst_xgb(train_bt, steps_val),
            "LSTM": fcst_lstm(train_bt, steps_val)
        }
        y_true = test_bt.values.astype(float)
        mse  = {k: (np.nan if np.isnan(v).all() else mean_squared_error(y_true, v)) for k,v in preds_val.items()}
        mape = {k: (np.nan if np.isnan(v).all() else safe_mape(y_true, v))          for k,v in preds_val.items()}

        # ----- Future (train on ALL data) -----
        preds_future: Dict[str, np.ndarray] = {
            "HW":   fcst_hw(ts, steps_future),
            "LR":   fcst_lr(ts, steps_future),
            "XGB":  fcst_xgb(ts, steps_future),
            "LSTM": fcst_lstm(ts, steps_future)
        }
        delta, _, _ = sampling_info(ts)
        if delta is None:
            delta = pd.Timedelta(hours=1)
        last_time = ts.index.max()
        future_index = [last_time + (i+1)*delta for i in range(steps_future)]

        # ----- Assemble output -----
        # History rows
        hist = history_slice(ts, HISTORY_HOURS)
        hist_df = pd.DataFrame({
            "Timestamp": hist.index,
            "Sensor Name": sensor,
            "Type": "historical",
            "Actual": hist.values,
            "HW": np.nan, "LR": np.nan, "XGB": np.nan, "LSTM": np.nan,
            "MSE HW": np.nan, "MSE LR": np.nan, "MSE XGB": np.nan, "MSE LSTM": np.nan,
            "MAPE HW": np.nan, "MAPE LR": np.nan, "MAPE XGB": np.nan, "MAPE LSTM": np.nan,
            "Latitude": lat, "Longitude": lon, "Formatted Address": formatted_addr
        })

        # Validation rows (aligned to real timestamps)
        val_df = pd.DataFrame({
            "Timestamp": test_bt.index,
            "Sensor Name": sensor,
            "Type": "validation_forecast",
            "Actual": y_true,
            "HW": preds_val["HW"],
            "LR": preds_val["LR"],
            "XGB": preds_val["XGB"],
            "LSTM": preds_val["LSTM"],
            "MSE HW": mse["HW"], "MSE LR": mse["LR"], "MSE XGB": mse["XGB"], "MSE LSTM": mse["LSTM"],
            "MAPE HW": mape["HW"], "MAPE LR": mape["LR"], "MAPE XGB": mape["XGB"], "MAPE LSTM": mape["LSTM"],
            "Latitude": lat, "Longitude": lon, "Formatted Address": formatted_addr
        })

        # Future rows (beyond last timestamp)
        fut_df = pd.DataFrame({
            "Timestamp": future_index,
            "Sensor Name": sensor,
            "Type": "future_forecast",
            "Actual": np.nan,
            "HW": preds_future["HW"],
            "LR": preds_future["LR"],
            "XGB": preds_future["XGB"],
            "LSTM": preds_future["LSTM"],
            "MSE HW": mse["HW"], "MSE LR": mse["LR"], "MSE XGB": mse["XGB"], "MSE LSTM": mse["LSTM"],
            "MAPE HW": mape["HW"], "MAPE LR": mape["LR"], "MAPE XGB": mape["XGB"], "MAPE LSTM": mape["LSTM"],
            "Latitude": lat, "Longitude": lon, "Formatted Address": formatted_addr
        })

        all_rows.extend([hist_df, val_df, fut_df])
        print(f"[done] {sensor}: {len(ts)} pts | val {steps_val} steps | future {steps_future} steps | {(time.time()-t0):.1f}s")

    if not all_rows:
        raise RuntimeError("No sensor produced output. Check data sufficiency.")

    result = pd.concat(all_rows, ignore_index=True)
    result["Timestamp"] = pd.to_datetime(result["Timestamp"])
    result.sort_values(["Sensor Name", "Timestamp", "Type"], inplace=True)
    result.to_csv(OUT_FILE, index=False)
    print(f"\nSaved ALL sensors to: {OUT_FILE}")

if __name__ == "__main__":
    main()

