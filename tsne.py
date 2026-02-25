import pandas as pd
import glob

# Step 1: Get a list of all your CSV files
csv_files = glob.glob('/home/campus.ncl.ac.uk/nsv53/Sneha/dashboard/*.csv')  # Change this path

# Step 2: Initialize an empty list to store DataFrames
dfs = []

# Step 3: Read and filter each CSV file
for file in csv_files:
    print(f'Merging file: {file}')  # Print which file is being merged
    df = pd.read_csv(file)
    
    # Step 4: Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Step 5: Get the current date and calculate the date 10 days ago
    last_10_days = pd.to_datetime('today') - pd.Timedelta(days=10)
    
    # Step 6: Filter the rows to keep only the last 10 days of data
    filtered_df = df[df['Timestamp'] >= last_10_days]
    
    # Step 7: Append the filtered DataFrame to the list
    dfs.append(filtered_df)

# Step 8: Merge all filtered DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)

# Step 9: Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_last_10_days.csv', index=False)

print('All files have been successfully merged with the last 10 days of data!')
