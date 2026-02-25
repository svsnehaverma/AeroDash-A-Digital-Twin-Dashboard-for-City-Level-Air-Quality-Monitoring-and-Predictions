import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderServiceError

# LSTM Model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

# LSTM Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=24):
        self.series = series.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

# Holt-Winters forecast
def forecast_hw(ts, steps=168):
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=24)
    fit = model.fit()
    return fit.forecast(steps)

# Geocoding setup
geolocator = Nominatim(user_agent="your_custom_app_name")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=3, error_wait_seconds=5.0)

def format_address(location):
    if not location:
        return None
    
    address = location.raw.get('address', {})

    road = address.get('road', '')
    neighbourhood = address.get('neighbourhood', '') or address.get('suburb', '')
    city = address.get('city', '') or address.get('town', '') or address.get('village', '')
    county = address.get('county', '')
    state = address.get('state', '')  # e.g., North East
    country = address.get('country', '')
    postcode = address.get('postcode', '')

    parts = [road, neighbourhood, city, county, state]

    # Insert postcode before country if postcode exists
    if postcode:
        parts.append(postcode)
    if country:
        parts.append(country)

    # Filter out empty parts and join by comma
    formatted_address = ', '.join([part for part in parts if part])

    return formatted_address

def safe_reverse_geocode(lat, lon):
    try:
        location = reverse((lat, lon))
        if location:
            return format_address(location)
        else:
            return None
    except GeocoderServiceError as e:
        print(f"Geocoding failed for ({lat}, {lon}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for ({lat}, {lon}): {e}")
        return None

# Read data
df = pd.read_csv("PM1 - PM1_till_19_Aug.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

results = []

sensor_ids = df['Sensor Name'].unique()

for sensor in sensor_ids:
    sensor_data = df[df['Sensor Name'] == sensor].copy()
    if sensor_data.shape[0] < 100:
        print(f"Skipping sensor {sensor} due to insufficient data")
        continue

    try:
        ts = sensor_data.set_index('Timestamp').resample('h').mean(numeric_only=True)['Value']
        ts = ts.ffill().bfill()

        # Normalize for ML models
        scaler = MinMaxScaler()
        ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()

        train_scaled = ts_scaled[:-168]
        test_scaled = ts_scaled[-168:]
        train_ts = ts[:-168]
        test_ts = ts[-168:]

        # Holt-Winters
        forecast_hw_vals = forecast_hw(train_ts, steps=168)

        # Linear Regression
        X_train = np.arange(len(train_scaled)).reshape(-1, 1)
        X_test = np.arange(len(train_scaled), len(train_scaled) + 168).reshape(-1, 1)
        model_lr = LinearRegression()
        model_lr.fit(X_train, train_scaled)
        forecast_lr_scaled = model_lr.predict(X_test)
        forecast_lr_vals = scaler.inverse_transform(forecast_lr_scaled.reshape(-1, 1)).flatten()

        # XGBoost
        model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model_xgb.fit(X_train, train_scaled)
        forecast_xgb_scaled = model_xgb.predict(X_test)
        forecast_xgb_vals = scaler.inverse_transform(forecast_xgb_scaled.reshape(-1, 1)).flatten()

        # LSTM
        seq_len = 24
        lstm_model = LSTMPredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
        train_dataset = TimeSeriesDataset(train_scaled, seq_len=seq_len)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        lstm_model.train()
        for epoch in range(30):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = lstm_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # LSTM prediction
        lstm_model.eval()
        test_inputs = list(train_scaled[-seq_len:])
        lstm_preds_scaled = []
        for _ in range(168):
            seq = torch.tensor(test_inputs[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                pred = lstm_model(seq).item()
            lstm_preds_scaled.append(pred)
            test_inputs.append(pred)
        forecast_lstm_vals = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()

        # Evaluation
        y_true = test_ts.values
        mse_hw = mean_squared_error(y_true, forecast_hw_vals)
        mse_lr = mean_squared_error(y_true, forecast_lr_vals)
        mse_xgb = mean_squared_error(y_true, forecast_xgb_vals)
        mse_lstm = mean_squared_error(y_true, forecast_lstm_vals)

        mape_hw = mean_absolute_percentage_error(y_true, forecast_hw_vals)
        mape_lr = mean_absolute_percentage_error(y_true, forecast_lr_vals)
        mape_xgb = mean_absolute_percentage_error(y_true, forecast_xgb_vals)
        mape_lstm = mean_absolute_percentage_error(y_true, forecast_lstm_vals)

        lat = sensor_data['Sensor Centroid Latitude'].iloc[0]
        lon = sensor_data['Sensor Centroid Longitude'].iloc[0]
        formatted_address = safe_reverse_geocode(lat, lon)

        # Append to results
        for i in range(168):
            results.append({
                'Timestamp': test_ts.index[i],
                'Sensor Name': sensor,
                'Holt-Winters': forecast_hw_vals[i],
                'Linear Regression': forecast_lr_vals[i],
                'XGBoost': forecast_xgb_vals[i],
                'LSTM': forecast_lstm_vals[i],
                'MSE Holt-Winters': mse_hw,
                'MSE Linear Regression': mse_lr,
                'MSE XGBoost': mse_xgb,
                'MSE LSTM': mse_lstm,
                'MAPE Holt-Winters': mape_hw,
                'MAPE Linear Regression': mape_lr,
                'MAPE XGBoost': mape_xgb,
                'MAPE LSTM': mape_lstm,
                'Latitude': lat,
                'Longitude': lon,
                'Formatted Address': formatted_address
            })

    except Exception as e:
        print(f"Error processing sensor {sensor}: {e}")

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("PM1_forecast_models_full_metrics.csv", index=False)
print("Saved to PM1_forecast_models_full_metrics.csv")

