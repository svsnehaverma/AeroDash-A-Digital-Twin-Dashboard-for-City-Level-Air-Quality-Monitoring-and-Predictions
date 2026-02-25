import requests
import csv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from io import StringIO

# --- SENSOR VARIABLES TO FETCH ---
variables = ["temp"]

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




import requests
import pandas as pd
import io

import requests
import pandas as pd
import io


import requests
import pandas as pd
import io

# --- GOOGLE SHEETS AUTHENTICATION ---
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    '/home/campus.ncl.ac.uk/nsv53/Downloads/openapi.json', scope
)
client = gspread.authorize(creds)

url = "https://api.v2.urbanobservatory.ac.uk/sensors/data/csv"

try:
    response = requests.get(url, params = {"variable": "Temperature", "last_n_days":5})
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(io.StringIO(response.text))

    # Print the first few rows of the DataFrame to verify
    print(df.head())

except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
except pd.errors.EmptyDataError:
    print("The received data was empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



'''


import requests
import pandas as pd
from io import StringIO

def fetch_no2_data(last_n_days=2, variable="temp"):
    """
    Fetches NO₂ sensor data for the specified number of days using the Urban Observatory API.
    
    Args:
        last_n_days (int): Number of days of data to retrieve.
        variable (str): Data variable to retrieve (e.g., "NO2").
    
    Returns:
        pandas.DataFrame: DataFrame containing the sensor data.
    """
    base_url = "https://newcastle.urbanobservatory.ac.uk/api/v1.1/sensors/data/csv/"
    params = {
        "last_n_days": last_n_days,
        "data_variable": variable
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raises error if HTTP request failed
    
    csv_text = response.text
    df = pd.read_csv(StringIO(csv_text))
    return df

if __name__ == "__main__":
    try:
        df = fetch_no2_data(2, "CO")
        print("Data retrieved successfully!")
        print(df.head())
    except Exception as e:
        print(f"Failed to retrieve data: {e}")
'''
