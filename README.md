# AeroDash: A Free and Open-Source Digital Twin Dashboard for City-Level Air Quality Monitoring and Prediction

AeroDash is an end-to-end data pipeline developed to support the research paper **“AeroDash”**, enabling automated ingestion of air quality data from the Urban Observatory, time-series forecasting using Holt–Winters models, and real-time visualization via Google Looker Studio dashboards.

The system integrates **Urban Observatory (UO) sensor data**, **Google Cloud Platform (GCP)**, **Google Sheets**, and **Looker Studio** to provide daily updates and short-term (72-hour) air quality predictions.

---
## 📌 Key Features

- Automated air quality data ingestion from **Urban Observatory**
- Secure data transfer and authentication using **Google Cloud Platform**
- Daily scheduled updates to **Google Sheets**
- 72-hour air quality forecasting using **Holt–Winters time-series models**
- Error analysis and validation of predictions
- Interactive dashboards built with **Google Looker Studio**
- Support for weekday/weekend and 1-minute resolution forecasting

---
## 📂 Repository Structure

---
## 🗂 File Descriptions

### 1. `Pulling_air_quality_data_from_UO.py`

- Fetches air quality sensor data from the **Urban Observatory API**
- Runs on **Google Cloud Platform**
- Authenticates using the service account key:
- Updates raw sensor data into **Google Sheets**
- Configured to run **once per day** (e.g., via Cloud Scheduler or cron job)
- The updated Google Sheet serves as the **live data source** for Looker Studio

---
### 2. `predictions_for_72_hours.py`

- Generates **72-hour air quality forecasts**
- Uses **Holt–Winters exponential smoothing**
- Supports:
- Weekday vs Weekend patterns
- 1-minute temporal resolution
- Outputs predictions to:
- The CSV is automatically synced to **Google Sheets**
- Predictions are visualized in Looker Studio

---
### 3. `predictions_error.py`

- Computes prediction accuracy and error metrics
- Supports validation against observed sensor values
- Used for:
- Model performance evaluation
- Error trend analysis
- Research reporting and benchmarking

---
### 4. `PM1_all_sensors_72h_forecasts_raw.csv`

- Contains raw 72-hour forecast outputs for all PM1 sensors
- Stored and maintained in **Google Sheets**
- Acts as the backend data source for Looker Studio dashboards

---

### 5. `HoltWinters_72h_Forecasts_WeekdayWeekend_1min`

- Looker Studio dashboard
- Visualizes:
- Short-term air quality forecasts
- Temporal trends
- Weekday vs Weekend variations
- Automatically refreshes as Google Sheets are updated

---
### 6. `HoltWinters_72h_Forecasts.zip`

- Archived prediction outputs
- Used for:
- Reproducibility
- Offline analysis
- Supplementary material for the AeroDash paper

---
### 7. `dashboard-457016-144084529a40.json`

- Google Cloud **service account credentials**
- Required for:
- Google Sheets API access
- Looker Studio data updates

---

## 🔄 System Workflow

---
## 📊 Looker Studio Integration

- Google Sheets acts as the **single source of truth**
- Dashboards auto-refresh once daily after data updates
- Visual analytics support:
  - Near-real-time monitoring
  - Short-term forecasting insights
  - Policy and research decision-making

---
## ⚙️ Requirements

- Python 3.8+
- Google Cloud Platform account
- Google Sheets API enabled
- Looker Studio access
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `statsmodels`
  - `google-api-python-client`
  - `gspread`

---
## 🔐 Security Notes

- Never commit `.json` service account keys to public repositories
- Use environment variables or secret managers where possible
- Restrict Google Sheets and API permissions to minimum required scope

---
## 📄 Citation

If you use this code or dashboard framework, please cite the **AeroDash paper** accordingly.

---

## 👤 Author

Sneha Verma<sup>a,b</sup>, Balendra V. S. Chauhan<sup>b</sup>, Xiang Xie<sup>b</sup>, Philip James<sup>c</sup>, Mohamad Kassem<sup>b</sup>

<sup>a</sup> School of Engineering and Computer Science, University of Sunderland, Sunderland, United Kingdom
<sup>b</sup> School of Engineering, Stephenson Building, Newcastle University, Newcastle upon Tyne, United Kingdom
<sup>c</sup> School of Computing, Urban Sciences Building, Newcastle University, Newcastle upon Tyne, United Kingdom

📧 Corresponding author:
Sneha Verma – sneha.verma@newcastle.ac.uk
; sneha.verma@sunderland.ac.uk

---
## 📬 Contact

For questions, collaboration, or reuse of this work, please contact the author via institutional email.
---

