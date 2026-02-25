'''
import requests
import csv
from io import StringIO

# API endpoint with parameters
url = "https://newcastle.urbanobservatory.ac.uk/api/v1.1/sensors/data/csv/"
params = {
    "last_n_days": 1,
    "data_variable": "NO2"
}

# Fetch data from the API
response = requests.get(url, params=params)
response.raise_for_status()

# Parse the CSV data
csv_text = response.text
csv_reader = csv.reader(StringIO(csv_text))
data = list(csv_reader)

# Display the fetched data
for row in data[1:4]:
    print(row)
'''
'''
import requests
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from io import StringIO

# --- API CONFIGURATION ---
url = "https://newcastle.urbanobservatory.ac.uk/api/v1.1/sensors/data/csv/"
params = {"last_n_days": 30, "data_variable": "NO2"}

# Fetch data from the API
response = requests.get(url, params=params)
response.raise_for_status()

# Parse the CSV data
csv_text = response.text
csv_reader = csv.reader(StringIO(csv_text))
data = list(csv_reader)

# --- GOOGLE SHEETS CONFIGURATION ---
# Define the scope and credentials
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('/home/campus.ncl.ac.uk/nsv53/Downloads/dashboard-457016-144084529a40.json', scope)

# Authenticate and open the spreadsheet
client = gspread.authorize(creds)
spreadsheet = client.open('NO2')  # Replace with your spreadsheet name
sheet = spreadsheet.sheet1  # Access the first sheet

# --- UPDATE GOOGLE SHEET ---
# Clear existing data
sheet.clear()

# Update with new data
sheet.update('A1', data)

print("Data updated successfully in Google Sheets.")
'''

import requests
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from io import StringIO

# --- API CONFIGURATION ---
url = "https://newcastle.urbanobservatory.ac.uk/api/v1.1/sensors/data/csv/"
params = {"last_n_days": 30, "data_variable": "Temperature"}

# Fetch data from the API
response = requests.get(url, params=params)
response.raise_for_status()

# Parse the CSV data
csv_text = response.text
csv_reader = csv.reader(StringIO(csv_text))
data = list(csv_reader)

# --- GOOGLE SHEETS CONFIGURATION ---
# Define the scope and credentials
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('/home/campus.ncl.ac.uk/nsv53/Downloads/openapi.json', scope)

# Authenticate and open the spreadsheet
client = gspread.authorize(creds)
spreadsheet = client.open('Temperature')  # Replace with your spreadsheet name
sheet = spreadsheet.sheet1  # Access the first sheet

# Clear existing data
sheet.clear()

# Update with new data
sheet.update('A1', data)

print("PM1 updated successfully in Google Sheets.")



