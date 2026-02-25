import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderServiceError

# ==============================
# CONFIG
# ==============================
input_file = "PM1 - PM1_till_19_Aug.csv"
output_dir = "HoltWinters_72h_Forecasts_WeekdayWeekend_1min"
os.makedirs(output_dir, exist_ok=True)

forecast_start = pd.to_datetime("2025-08-19 10:29:00")

forecast_horizon_hours = 72
freq = "1min"                               # <<< keep SAME sampling as original data
steps = forecast_horizon_hours * 60          # 72h at 1-min = 4320 points
forecast_index = pd.date_range(forecast_start, periods=steps, freq=freq)
forecast_end = forecast_index[-1]

# For 1-minute data:
# daily seasonality = 1440 minutes
seasonal_periods_default = 1440

MIN_TRAIN_POINTS = 200  # keep a basic minimum
MIN_POINTS_FOR_SEASONAL = 2 * seasonal_periods_default  # rule of thumb

# ==============================
# HELPER FUNCTIONS
# ==============================
def forecast_hw(ts, steps, seasonal_periods):
    """
    Fit Holt-Winters model and forecast 'steps' ahead.
    Uses multiplicative seasonality to help preserve amplitude (less "flat" forecasts).
    Falls back if the seasonal model cannot be fit.
    """
    # Drop any NaNs just in case
    ts = ts.dropna()

    # Try seasonal model first
    try:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="mul",  # often helps when forecasts look low/flat
            seasonal_periods=seasonal_periods
        )
        fit = model.fit(optimized=True, use_brute=True)
        return fit.forecast(steps), fit, {"trend": "add", "seasonal": "mul", "seasonal_periods": seasonal_periods}
    except Exception:
        pass

    # Fallback: additive seasonality
    try:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods
        )
        fit = model.fit(optimized=True, use_brute=True)
        return fit.forecast(steps), fit, {"trend": "add", "seasonal": "add", "seasonal_periods": seasonal_periods}
    except Exception:
        pass

    # Final fallback: no seasonality
    model = ExponentialSmoothing(ts, trend="add", seasonal=None)
    fit = model.fit(optimized=True, use_brute=True)
    return fit.forecast(steps), fit, {"trend": "add", "seasonal": None, "seasonal_periods": None}


# Geocoder for location metadata
geolocator = Nominatim(user_agent="aerodash_app")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=3)

# Cache to avoid repeated reverse-geocode calls (big speed-up for many sensors)
_address_cache = {}

def safe_reverse_geocode(lat, lon):
    # Round key to reduce duplicates from tiny float differences
    key = (round(float(lat), 5), round(float(lon), 5))
    if key in _address_cache:
        return _address_cache[key]

    try:
        location = reverse((key[0], key[1]))
        if not location:
            _address_cache[key] = "Unknown Location"
            return _address_cache[key]

        address = location.raw.get("address", {})
        parts = [address.get(k, "") for k in ["road", "neighbourhood", "city", "county", "state", "country"]]
        text = ", ".join([p for p in parts if p]).strip()
        _address_cache[key] = text if text else "Unknown Location"
        return _address_cache[key]

    except GeocoderServiceError:
        _address_cache[key] = "Service Error"
        return _address_cache[key]
    except Exception:
        _address_cache[key] = "Geocoding Error"
        return _address_cache[key]


def get_weekend_spans(index):
    """
    Given a DateTimeIndex (historical + forecast),
    return list of (start, end) tuples for contiguous weekend periods.
    Weekend = Saturday (5) or Sunday (6).
    """
    index = index.sort_values()
    weekend_mask = index.weekday >= 5

    spans = []
    current_start = None

    for t, is_weekend in zip(index, weekend_mask):
        if is_weekend and current_start is None:
            current_start = t
        elif (not is_weekend) and (current_start is not None):
            spans.append((current_start, t))
            current_start = None

    if current_start is not None:
        # close the last span one step after last timestamp
        step = index[-1] - index[-2] if len(index) > 1 else pd.Timedelta(minutes=1)
        spans.append((current_start, index[-1] + step))

    return spans


def format_weekend_dates(combined_index, max_dates=20):
    weekend_dates = sorted({ts.date() for ts in combined_index if ts.weekday() >= 5})
    if not weekend_dates:
        return "", []

    truncated = False
    if len(weekend_dates) > max_dates:
        weekend_dates = weekend_dates[:max_dates]
        truncated = True

    formatted_dates = [pd.Timestamp(d).strftime("%d %b") for d in weekend_dates]
    text = ", ".join(formatted_dates)
    if truncated:
        text += ", ..."
    return text, weekend_dates


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(input_file)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp", "Sensor Name", "Value"])

sensor_ids = df["Sensor Name"].unique()
results = []

# ==============================
# PROCESS EACH SENSOR
# ==============================
for sensor in sensor_ids:
    sensor_data = df[df["Sensor Name"] == sensor].copy()
    sensor_data = sensor_data.dropna(subset=["Value"])
    if sensor_data.shape[0] < MIN_TRAIN_POINTS:
        print(f"Skipping {sensor} (insufficient raw rows)")
        continue

    try:
        # ==============================
        # KEEP ORIGINAL SAMPLING RATE (NO AVERAGING!)
        # ==============================
        sensor_data = sensor_data.sort_values("Timestamp")
        ts = sensor_data.set_index("Timestamp")["Value"].astype(float)

        # Force to 1-min frequency without averaging:
        ts = ts.asfreq(freq)

        # Fill missing timestamps (no smoothing beyond fill)
        ts = ts.ffill().bfill()

        # Train/Test split using exact forecast_index alignment
        train_ts = ts[ts.index < forecast_start]
        if train_ts.empty or len(train_ts) < MIN_TRAIN_POINTS:
            print(f"Skipping {sensor} (not enough training points before {forecast_start})")
            continue

        test_ts = ts.reindex(forecast_index)  # aligns exactly

        # ==============================
        # WEEKDAY / WEEKEND ANALYSIS (on train)
        # ==============================
        dow_means = train_ts.groupby(train_ts.index.dayofweek).mean()

        weekday_mean = np.nan
        weekend_mean = np.nan
        try:
            weekday_mean = dow_means[[0, 1, 2, 3, 4]].mean()
        except KeyError:
            pass
        try:
            weekend_mean = dow_means[[5, 6]].mean()
        except KeyError:
            pass

        # ==============================
        # CHOOSE SEASONAL PERIODS SAFELY
        # ==============================
        seasonal_periods = seasonal_periods_default
        if len(train_ts) < MIN_POINTS_FOR_SEASONAL:
            # not enough history for daily seasonality -> fallback to smaller or none
            # try 6-hour seasonality (360 minutes) if possible, else no seasonal
            if len(train_ts) >= 2 * 360:
                seasonal_periods = 360
            else:
                seasonal_periods = None

        # Forecast
        if seasonal_periods is None:
            # force no seasonality
            fc, fit, cfg = forecast_hw(train_ts, steps=steps, seasonal_periods=2)  # seasonal_periods ignored in final fallback
        else:
            fc, fit, cfg = forecast_hw(train_ts, steps=steps, seasonal_periods=seasonal_periods)

        forecast_hw_vals = pd.Series(fc.values if hasattr(fc, "values") else np.asarray(fc), index=forecast_index)

        # Evaluation (only where actuals exist)
        y_true = test_ts.values.astype(float)
        y_pred = forecast_hw_vals.values.astype(float)
        mask = ~np.isnan(y_true)

        if mask.any():
            mse_hw = mean_squared_error(y_true[mask], y_pred[mask])
            mape_hw = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            mse_hw = np.nan
            mape_hw = np.nan

        # Location metadata
        lat = sensor_data["Sensor Centroid Latitude"].iloc[0] if "Sensor Centroid Latitude" in sensor_data.columns else np.nan
        lon = sensor_data["Sensor Centroid Longitude"].iloc[0] if "Sensor Centroid Longitude" in sensor_data.columns else np.nan
        formatted_address = safe_reverse_geocode(lat, lon) if np.isfinite(lat) and np.isfinite(lon) else "Unknown Location"

        # Save result rows (4320 rows per sensor for 72h at 1-min)
        for t, v in forecast_hw_vals.items():
            results.append({
                "Timestamp": t,
                "Sensor Name": sensor,
                "Forecast (HW)": float(v),
                "Latitude": lat,
                "Longitude": lon,
                "Formatted Address": formatted_address,
                "MSE Holt-Winters": mse_hw,
                "MAPE Holt-Winters": mape_hw,
                "Weekday Mean PM1 (train)": weekday_mean,
                "Weekend Mean PM1 (train)": weekend_mean,
                "HW trend": cfg.get("trend"),
                "HW seasonal": cfg.get("seasonal"),
                "HW seasonal_periods": cfg.get("seasonal_periods"),
            })

        # ==============================
        # WEEKEND DATES (FOR PRINT & PLOT)
        # ==============================
        combined_index = train_ts.index.union(forecast_index)
        weekend_dates_text, weekend_dates_list = format_weekend_dates(combined_index)

        print(f"\nSensor: {sensor}")
        print(f"HW config used: {cfg}")
        if weekend_dates_list:
            printable_dates = ", ".join(pd.Timestamp(d).strftime("%d %b") for d in weekend_dates_list)
            print(f"Weekend days in historical+forecast window: {printable_dates}")
        else:
            print("No weekend days in historical+forecast window.")

        # ==============================
        # PLOT RESULTS
        # ==============================
        plt.figure(figsize=(12, 6))

        # Historical
        plt.plot(train_ts.index, train_ts.values, label="Historical (Actual)", linewidth=1.5)

        # Forecast
        plt.plot(forecast_index, forecast_hw_vals.values, "--", label="Holt–Winters Forecast", linewidth=1.2)

        # Weekend shading
        weekend_spans = get_weekend_spans(combined_index)
        for ws, we in weekend_spans:
            plt.axvspan(ws, we, alpha=0.2, zorder=0)

        # Forecast window shading
        plt.axvspan(forecast_start, forecast_end, alpha=0.15, label="Forecast Period")

        # Metrics text
        metrics_lines = []
        if not np.isnan(mse_hw):
            metrics_lines.append(f"MSE={mse_hw:.4f}")
        if not np.isnan(mape_hw):
            metrics_lines.append(f"MAPE={mape_hw * 100:.2f}%")
        if not np.isnan(weekday_mean):
            metrics_lines.append(f"Weekday mean={weekday_mean:.2f} µg/m³")
        if not np.isnan(weekend_mean):
            metrics_lines.append(f"Weekend mean={weekend_mean:.2f} µg/m³")
        if weekend_dates_text:
            metrics_lines.append("Weekends:")
            metrics_lines.append(weekend_dates_text)

        if metrics_lines:
            plt.gcf().text(
                0.70, 0.20,
                "\n".join(metrics_lines),
                fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.75)
            )

        plt.title(
            f"PM₁ Forecast (Holt–Winters) — 72h ahead @ {freq}\n"
            f"{formatted_address}\n"
            f"Sensor: {sensor}\n"
            f"Weekend shading shown",
            fontsize=11,
            wrap=True
        )
        plt.xlabel("Timestamp (UTC)")
        plt.ylabel("PM₁ Concentration (µg/m³)")
        plt.legend(loc="upper left", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        outfile = os.path.join(output_dir, f"{sensor}_HW_72h_{freq}_weekday_weekend.png")
        plt.savefig(outfile, dpi=250)
        plt.close()

        print(f"Saved plot: {outfile}")

    except Exception as e:
        print(f"Error processing {sensor}: {e}")

# ==============================
# SAVE RESULTS
# ==============================
df_results = pd.DataFrame(results)
csv_path = os.path.join(output_dir, f"HoltWinters_72h_Forecasts_{freq}_weekday_weekend.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nForecast data saved to {csv_path}")

