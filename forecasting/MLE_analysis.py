import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter
import warnings

#MLE analysis for every station phys, savgol filter, and raw data
# tests MLE over ~10 year period 

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
STATIONS = ['MAHI', 'GISB', 'KOKO', 'CKID', 'PAWA', 'CNST', 'PORA']
METHODS = ['model', 'raw', 'savgol']
WINDOWS = [2016.6627,  2016.9458, 2018.00, 2021.1886, 2024.9233]

PHASE_CONFIG = {
    'GISB': {'model': {'tau': 90, 'm': 3, 'sigma': 7.87}, 'raw': {'tau': 17, 'm': 3, 'sigma': 7.64}, 'savgol': {'tau': 73, 'm': 3, 'sigma': 7.64}},
    'KOKO': {'model': {'tau': 62, 'm': 3, 'sigma': 5.57}, 'raw': {'tau': 41, 'm': 3, 'sigma': 5.30}, 'savgol': {'tau': 41, 'm': 3, 'sigma': 5.30}},
    'MAHI': {'model': {'tau': 74, 'm': 3, 'sigma': 9.16}, 'raw': {'tau': 36, 'm': 3, 'sigma': 9.08}, 'savgol': {'tau': 50, 'm': 3, 'sigma': 9.08}},
    'PAWA': {'model': {'tau': 91, 'm': 3, 'sigma': 9.55}, 'raw': {'tau': 27, 'm': 3, 'sigma': 9.13}, 'savgol': {'tau': 42, 'm': 3, 'sigma': 9.13}},
    'CKID': {'model': {'tau': 77, 'm': 3, 'sigma': 5.62}, 'raw': {'tau': 13, 'm': 3, 'sigma': 5.09}, 'savgol': {'tau': 50, 'm': 3, 'sigma': 5.09}},
    'CNST': {'model': {'tau': 81, 'm': 3, 'sigma': 8.50}, 'raw': {'tau': 36, 'm': 3, 'sigma': 8.30}, 'savgol': {'tau': 52, 'm': 3, 'sigma': 8.30}},
    'PORA': {'model': {'tau': 88, 'm': 3, 'sigma': 8.30}, 'raw': {'tau': 7,  'm': 3, 'sigma': 7.78}, 'savgol': {'tau': 53, 'm': 3, 'sigma': 7.78}}
}

def load_data_all_methods(station):
    # Load model and raw files
    model_df = pd.read_csv(f'phys_fits/{station}_model.csv')
    raw_df = pd.read_csv(f'detrended_data/{station}_clean.csv')
    
    x = raw_df.iloc[:, 0].values
    y_raw = -raw_df.iloc[:, 1].values - raw_df.iloc[0, 1]
    p_fit = np.polyfit(x, y_raw, 1)
    y_raw = y_raw - np.polyval(p_fit, x)
    
    y_savgol = savgol_filter(y_raw, window_length=51, polyorder=3)
    
    y_model = model_df.iloc[:, 1].values
    x_model = model_df.iloc[:, 0].values
    
    return {'model': (x_model, y_model), 'raw': (x, y_raw), 'savgol': (x, y_savgol)}

def delay_vectors(data, m, tau):
    n = len(data)
    if n <= (m - 1) * tau: return None
    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        Y[:, i] = data[(m - 1 - i) * tau : n - i * tau]
    return Y

def calculate_mle(Y, dt, m, tau):
    if Y is None or len(Y) < 1000: return np.nan 
    Y_norm = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    iterations, theiler = 30, int(m * tau)
    nbrs = NearestNeighbors(n_neighbors=theiler + 5).fit(Y_norm)
    distances, indices = nbrs.kneighbors(Y_norm)
    divergence = []
    for i in range(len(Y_norm) - iterations):
        for neighbor_idx, d0 in zip(indices[i][1:], distances[i][1:]):
            if abs(neighbor_idx - i) > theiler and d0 > 1e-6:
                target_idx, d_start = neighbor_idx, d0
                break
        else: continue
        if (target_idx + iterations) < len(Y_norm):
            d_future = np.linalg.norm(Y_norm[i + iterations] - Y_norm[target_idx + iterations])
            divergence.append(np.log(d_future / d_start))
    if not divergence: return np.nan
    lambda_max = np.mean(divergence) / (iterations * dt)
    return (1 / lambda_max) * 365.25 if lambda_max > 0 else np.nan

if __name__ == '__main__':
    header = f"{'Station':<8} | {'Method':<7} | {'thru 2016':<12} | {'thru 2018':<12} | {'thru 2021':<12} | {'thru 2026'}"
    print(header)
    print("-" * len(header))

    for station in STATIONS:
        data_dict = load_data_all_methods(station)
        
        for method in METHODS:
            x, y = data_dict[method]
            dt = x[1] - x[0]
            cfg = PHASE_CONFIG[station][method]
            horizons = []
            
            for end_date in WINDOWS:
                mask = (x <= end_date)
                Y = delay_vectors(y[mask], cfg['m'], cfg['tau'])
                t_hor = calculate_mle(Y, dt, cfg['m'], cfg['tau'])
                horizons.append(f"{t_hor:>10.1f} d" if not np.isnan(t_hor) else "   Insuff.  ")
                
            print(f"{station:<8} | {method:<7} | {' | '.join(horizons)}")