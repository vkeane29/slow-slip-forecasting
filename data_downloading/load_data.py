import pandas as pd
import requests
import io
from tqdm import tqdm

# downloads east disp timeseries data from geonet website

sensors = ['GISB', 'MAHI', 'KOKO', 'PAWA', 'CKID', 'CNST', 'PORA']
all_data = []

def get_east_displ(site_id, start_year=2002):
    base_url = "https://tilde.geonet.org.nz/v4/data/gnss"
    start_date = f"{start_year}-01-01"
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    url = f"{base_url}/{site_id}/displacement/nil/1d/east/{start_date}/{end_date}"
    headers = {"Accept": "text/csv"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        df = df[['timestamp', 'value']].rename(columns={'value': site_id})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    else:
        print(f"Skipping {site_id}: No data found or error {response.status_code}")
        return None


for sensor in tqdm(sensors, desc="Fetching East disp data"):
    df = get_east_displ(sensor)
    if df is not None:
        all_data.append(df)

# Merge sensors into one csv
if all_data:
    combined_df = pd.concat(all_data, axis=1)
    combined_df.to_csv("download_data/geonet_east_displacement_2002_2026.csv")
    print("Saved to geonet_east_displacement_2002_2026.csv")