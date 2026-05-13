import pandas as pd
import numpy as np

# takes geonet_east_disp file and saves separate station .csv time series 
# output: {station name}_data.csv with time (decimal year), displ (m), displ (mm)

def to_decimal_year(date):
    # geonet data time is UTC
    year = date.year
    start_of_year = pd.Timestamp(year=year, month=1, day=1, tz='UTC')
    end_of_year = pd.Timestamp(year=year + 1, month=1, day=1, tz='UTC')
    return year + (date - start_of_year) / (end_of_year - start_of_year)

# Load combined file
input_file = "download_data/geonet_east_displacement_2002_2026.csv"
combined_df = pd.read_csv(input_file)

timestamp_col = combined_df.columns[0]
combined_df[timestamp_col] = pd.to_datetime(combined_df[timestamp_col], utc=True)

stations = [col for col in combined_df.columns if col != timestamp_col]

print(f"Processing stations: {stations}")

for station in stations:
    # filter for station and drop empty rows
    station_df = combined_df[[timestamp_col, station]].dropna()
    if not station_df.empty:
        output_df = pd.DataFrame()
        output_df['Time (decimal year)'] = station_df[timestamp_col].apply(to_decimal_year)
        output_df['displacement (m)'] = station_df[station]
        output_df['displacement (mm)'] = station_df[station] * 1000
        
        output_filename = f"timeser/{station}_data.csv"
        output_df.to_csv(output_filename, index=False)
        print(f"Created {output_filename}")
    else:
        print(f"Skipping {station}: No data.")