import pandas as pd

# === Load the metrics file ===
metrics_file = "PM1_forecast_models_full_metrics.csv"
df = pd.read_csv(metrics_file)

# === Define sensors of interest ===
selected_sensors = [
    "PER_AIRMON_MESH300245",
    "PER_AIRMON_MESH1915150",
    "PER_AIRMON_MESH303245",
    "PER_AIRMON_MONITOR1056100"
]

# === Filter dataset ===
df = df[df["Sensor Name"].isin(selected_sensors)]

# === Compute mean MAPE per sensor and per model ===
summary = df.groupby("Sensor Name").agg({
    "MAPE Holt-Winters": "mean",
    "MAPE Linear Regression": "mean",
    "MAPE XGBoost": "mean",
    "MAPE LSTM": "mean"
}).reset_index()

# === Convert to percentage if values < 1 ===
for col in summary.columns:
    if "MAPE" in col:
        summary[col] = summary[col].apply(lambda x: x * 100 if x < 1 else x)

# === Round for readability ===
summary = summary.round(3)

# === Print per-sensor table ===
print("\n================ Mean MAPE Summary (Selected Sensors) ================\n")
print(summary.to_string(index=False))
print("\n======================================================================\n")

# === Compute global average MAPE across all selected sensors ===
avg_summary = summary.drop(columns=["Sensor Name"]).mean().to_dict()

print("================ Global Average MAPE Across Selected Sensors ================\n")
for model, value in avg_summary.items():
    print(f"{model:<30}: {value:.2f}%")
print("======================================================================\n")

# === Save results ===
summary.to_csv("Selected_Sensors_MAPE_Summary.csv", index=False)
print(" Summary saved to: Selected_Sensors_MAPE_Summary.csv")

