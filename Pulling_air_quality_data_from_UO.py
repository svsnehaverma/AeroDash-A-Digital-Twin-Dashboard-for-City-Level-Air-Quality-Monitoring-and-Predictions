import requests
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from io import StringIO

# --- SENSOR VARIABLES TO FETCH ---
variables = ["CO", "NOx", "NO2", "O3", "Particle Count", "PM1", "PM2.5", "PM10", "PM 4"]

# --- GOOGLE SHEETS AUTHENTICATION ---
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    '/home/campus.ncl.ac.uk/nsv53/Downloads/dashboard-457016-144084529a40.json', scope
)
client = gspread.authorize(creds)

# --- API URL ---
url = "https://newcastle.urbanobservatory.ac.uk/api/v1.1/sensors/data/csv/"

# --- PROCESS EACH VARIABLE ---
for variable in variables:
    try:
        print(f"Fetching data for {variable}...")

        # Request data from API
        params = {"last_n_days": 30, "data_variable": variable}
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse CSV data
        csv_reader = csv.reader(StringIO(response.text))
        data = list(csv_reader)

        # Open or create the spreadsheet named after the variable
        try:
            spreadsheet = client.open(variable)
        except gspread.exceptions.SpreadsheetNotFound:
            spreadsheet = client.create(variable)

        sheet = spreadsheet.sheet1
        sheet.clear()  # Clear previous data
        sheet.update('A1', data)  # Upload new data

        print(f"{variable} data updated successfully in Google Sheets.")

    except Exception as e:
        print(f"Failed to update {variable}: {e}")

