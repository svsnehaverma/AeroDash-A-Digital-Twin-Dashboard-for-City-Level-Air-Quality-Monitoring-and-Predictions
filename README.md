# AeroDash-A-Digital-Twin-Dashboard-for-City-Level-Air-Quality-Monitoring-and-Predictions
This project integrates real-time data-fusion, hybrid ML-based forecasting, and interactive visualisation within a modular and scalable architecture that supports transparent and customisable dashboards. 

This repository accompanies the AeroDash paper and provides the full data ingestion, processing, forecasting, and visualisation pipeline for near-real-time air quality monitoring and prediction using data from the Urban Observatory (UO).

The system integrates:

Automated data ingestion from the Urban Observatory via Google Cloud Platform

Daily updates to Google Sheets

Interactive visualisation and knowledge dissemination through Google Looker Studio

72-hour air quality forecasting using Holt-Winters–based models

Forecast evaluation and error analysis

Repository Structure
.
├── Pulling_air_quality_data_from_UO.py
├── predictions_for_72_hours.py
├── predictions_error.py
├── HoltWinters_72h_Forecasts.zip
├── dashboard-457016-144084529a40.json
└── README.md

System Overview
1. Data Ingestion (Urban Observatory → Google Cloud)

Pulling_air_quality_data_from_UO.py

Pulls air quality sensor data (e.g., PM₁, PM₂.₅, NO₂, CO) from the Urban Observatory API

Runs on Google Cloud Platform

Updates a linked Google Sheet once per day

The Google Sheet serves as the live data source for Looker Studio

Authentication

Uses a Google Cloud service account:

dashboard-457016-144084529a40.json

2. Data Visualisation (Google Sheets → Looker Studio)

The updated Google Sheet is connected to Google Looker Studio

Enables:

Near-real-time air quality monitoring

Sensor-level and temporal analysis

Public-facing and research dashboards

3. 72-Hour Forecasting Pipeline
Forecast Generation

predictions_for_72_hours.py

Generates 72-hour forecasts for PM₁ using Holt-Winters–based time-series models

Incorporates weekday/weekend patterns and 1-minute temporal resolution

Outputs predictions to:

PM1_all_sensors_72h_forecasts_raw.csv

The CSV is automatically uploaded to Google Sheets, enabling Looker Studio to visualise forecasts

Forecast Evaluation

predictions_error.py

Computes forecast error metrics

Supports performance assessment and model validation

4. Looker Studio Forecast Dashboard

Dashboard name:
HoltWinters_72h_Forecasts_WeekdayWeekend_1min

Displays:

Forecasted PM₁ concentrations

Temporal trends over the next 72 hours

Knowledge extraction from model outputs for decision-making

5. Forecast Outputs

HoltWinters_72h_Forecasts.zip

Contains exported forecast results

Useful for offline analysis, replication, or archival

Automated Workflow

Daily

Urban Observatory data pulled via Pulling_air_quality_data_from_UO.py

Google Sheet updated on GCP

Looker Studio dashboard refreshes automatically

On Demand / Scheduled

Run predictions_for_72_hours.py to generate new 72-hour forecasts

Upload predictions to Google Sheets

Forecasts visualised in Looker Studio

Evaluation

Run predictions_error.py for accuracy and performance analysis

Requirements

Python ≥ 3.8

Google Cloud Platform account

Google Sheets API enabled

Looker Studio dashboard access

Required Python packages:

pandas

numpy

statsmodels

gspread

google-auth

requests

Reproducibility Notes

Ensure the service account JSON key is correctly referenced

Update Google Sheet IDs and Looker Studio data sources if deploying in a new environment

Sensor availability and Urban Observatory endpoints may evolve over time


For questions, collaborations, or issues, please contact the authors via GitHub or the corresponding author listed in the paper.
