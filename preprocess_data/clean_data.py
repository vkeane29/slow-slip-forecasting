import pandas as pd
import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
from tqdm import tqdm 

# calculates noise of each station and outputs detrended files 
# {station_name}_clean.csv including time (decimal year), disp (mm)

STATIONS = ['GISB', 'KOKO', 'MAHI', 'PAWA', 'CKID', 'CNST', 'PORA']
def seasonal_func(t, a1, p1, a2, p2):
    # Annual and semi-annual cycles
    return a1 * np.sin(2 * np.pi * t + p1) + a2 * np.sin(4 * np.pi * t + p2)

def analyze_and_clean(station):
    # Load raw data
    df = pd.read_csv(f"timeser/{station}_data.csv")
    t = df['Time (decimal year)'].values
    y = df['displacement (mm)'].values

    # linear detrend
    coeffs = np.polyfit(t, y, 1)
    detrended = y - np.polyval(coeffs, t)

    # remove seasonal effects
    try:
        popt, _ = curve_fit(seasonal_func, t, detrended)
        seasonal_removed = detrended - seasonal_func(t, *popt)
    except Exception as e:
        print(f"Warning: Seasonal fit failed for {station}, using detrended only.")
        seasonal_removed = detrended

    # Noise Analysis (Lomb-Scargle)
    duration = t.max() - t.min()
    freqs = np.linspace(1/duration, 182, 1000)
    angular_freqs = 2 * np.pi * freqs
    pgram = lombscargle(t, seasonal_removed, angular_freqs, normalize=True)
    
    alpha_fit = np.polyfit(np.log(freqs), np.log(pgram), 1)
    alpha = -alpha_fit[0]
    rms = np.sqrt(np.mean(seasonal_removed**2))

    # Save cleaned CSV 
    out_df = pd.DataFrame({'Time': t, 'Disp_mm': seasonal_removed})
    out_df.to_csv(f"detrended_data/{station}_clean.csv", index=False)
    return t, y, detrended, alpha, rms

results_summary = []

for station in tqdm(STATIONS, desc="Processing Stations"):
    try:
        t, raw, detrend, alpha, rms = analyze_and_clean(station)        
        results_summary.append({
            'Station': station,
            'Noise_RMS': rms,
            'Alpha': alpha
        })
    except FileNotFoundError:
        print(f"Error: Data file for {station} not found.")

# final summary table
print("\n" + "="*40)
print(f"{'STATION':<10} | {'Noise_RMS (mm)':<10} | {'ALPHA':<10}")
print("-" * 40)
for res in results_summary:
    print(f"{res['Station']:<10} | {res['Noise_RMS']:<10.2f} | {res['Alpha']:<10.2f}")
print("="*40)