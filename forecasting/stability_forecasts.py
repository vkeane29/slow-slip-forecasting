import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import warnings

# checks forecast convergence and stability
# runs forecasts for each sation on multiple times
# checks for convergence and correct rejection/false positive event prediction


warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. USER CONTROL PANEL ---
STATIONS = ['MAHI', 'GISB', 'KOKO', 'CKID', 'PAWA', 'CNST', 'PORA'] 

TARGET_DATES = {
    '2016_a': 2016.6627,
    '2016_b': 2016.9458,
    '2018': 2018.0000,
    '2019': 2019.0000,
    '2021': 2021.1886,
    '2024': 2024.9233,
    '2023': 2023.15,
    '2010': 2010.2,
    '2017.5': 2017.5,

}

ENSEMBLE_SAMPLES = 10  # Number of jittered runs to test reliability
JITTER_DAYS = 2        # Jitter the start date by +/- 2 days

HORIZON_TEST_DAYS = 17.5 # Evaluate forecast starting ~2.5 weeks out
FORECAST_DURATION_YRS = 0.25   
EVALUATION_WINDOW_DAYS = 30     

PHASE_CONFIG = {
    'GISB': {'model': {'tau': 90, 'm': 3, 'sigma': 7.87}, 'raw': {'tau': 17, 'm': 3, 'sigma': 7.64}, 'savgol': {'tau': 73, 'm': 3, 'sigma': 7.64}},
    'KOKO': {'model': {'tau': 62, 'm': 3, 'sigma': 5.57}, 'raw': {'tau': 41, 'm': 3, 'sigma': 5.30}, 'savgol': {'tau': 41, 'm': 3, 'sigma': 5.30}},
    'MAHI': {'model': {'tau': 74, 'm': 3, 'sigma': 9.16}, 'raw': {'tau': 36, 'm': 3, 'sigma': 9.08}, 'savgol': {'tau': 50, 'm': 3, 'sigma': 9.08}},
    'PAWA': {'model': {'tau': 91, 'm': 3, 'sigma': 9.55}, 'raw': {'tau': 27, 'm': 3, 'sigma': 9.13}, 'savgol': {'tau': 42, 'm': 3, 'sigma': 9.13}},
    'CKID': {'model': {'tau': 77, 'm': 3, 'sigma': 5.62}, 'raw': {'tau': 13, 'm': 3, 'sigma': 5.09}, 'savgol': {'tau': 50, 'm': 3, 'sigma': 5.09}},
    'CNST': {'model': {'tau': 81, 'm': 3, 'sigma': 8.50}, 'raw': {'tau': 36, 'm': 3, 'sigma': 8.30}, 'savgol': {'tau': 52, 'm': 3, 'sigma': 8.30}},
    'PORA': {'model': {'tau': 88, 'm': 3, 'sigma': 8.30}, 'raw': {'tau': 7,  'm': 3, 'sigma': 7.78}, 'savgol': {'tau': 53, 'm': 3, 'sigma': 7.78}}
}


def load_data(station):
    model_df = pd.read_csv(f'phys_fits/{station}_model.csv')
    raw_df = pd.read_csv(f'detrended_data/{station}_clean.csv')
    x_p, y_p = model_df.iloc[:, 0].values, model_df.iloc[:, 1].values
    x_r = raw_df.iloc[:, 0].values
    y_r = -raw_df.iloc[:, 1].values - raw_df.iloc[0, 1]
    p_fit = np.polyfit(x_r, y_r, 1)
    y_r = y_r - np.polyval(p_fit, x_r)
    y_s = savgol_filter(y_r, window_length=31, polyorder=2)
    return x_p, y_p, x_r, y_r, y_s

def delay_vectors(data, m, tau):
    n = len(data)
    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        Y[:, i] = data[(m - 1 - i) * tau : n - i * tau]
    return Y

def calculate_mle(Y, dt, m, tau):
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    Y_std[Y_std == 0] = 1 
    Y_norm = (Y - Y_mean) / Y_std

    iterations = 30
    theiler_window = int(m * tau)  
    
    nbrs = NearestNeighbors(n_neighbors=theiler_window + 5).fit(Y_norm)
    distances, indices = nbrs.kneighbors(Y_norm)
    
    divergence = []
    for i in range(len(Y_norm) - iterations):
        valid_idx = -1
        for neighbor_idx, d0 in zip(indices[i][1:], distances[i][1:]):
            if abs(neighbor_idx - i) > theiler_window:
                if d0 > 1e-6:
                    target_idx = neighbor_idx
                    valid_idx = neighbor_idx
                    d_start = d0
                    break
        
        if valid_idx == -1 or (target_idx + iterations) >= len(Y_norm):
            continue
            
        d_future = np.linalg.norm(Y_norm[i + iterations] - Y_norm[target_idx + iterations])
        divergence.append(np.log(d_future / d_start))
        
    if not divergence: return 0, 365.25
    lambda_max = np.mean(divergence) / (iterations * dt)
    t_hor = (1 / lambda_max) * 365.25 if lambda_max > 0 else 365.25
    return lambda_max, t_hor

def find_optimal_k(Y):
    K_RANGE = np.arange(60, 200, 10)
    val_size = 10

    train_Y, val_Y = Y[:-val_size], Y[-val_size:]
    best_k, min_err = 60, float('inf')
    
    ridge_model = Ridge(alpha=0.1, fit_intercept=False)
    h_x, h_x1 = train_Y[:-1], train_Y[1:]
    nbrs = NearestNeighbors(n_neighbors=max(K_RANGE)).fit(h_x) 

    for k in K_RANGE:
        errors = []
        for j in range(val_size):
            curr = val_Y[j]
            d, idx = nbrs.kneighbors(curr.reshape(1, -1), n_neighbors=k)
            
            dm = d[0].max() * 1.01
            if dm == 0: dm = 1e-6 # prevent division by zero
            w = (1 - (d[0]/dm)**3)**3
            W = np.diag(w)
            
            X_a = np.column_stack((h_x[idx[0]], np.ones(k)))
            
            ridge_model.fit(W @ X_a, W @ h_x1[idx[0]])
            pred = np.append(curr, 1) @ ridge_model.coef_.T
            errors.append((pred[0] - val_Y[j, 0])**2)
        
        err = np.sqrt(np.mean(errors))
        if err < min_err: 
            min_err = err
            best_k = k
            
    return best_k

def knn_forecast(Y, steps, k):
    h_x, h_x1 = Y[:-1], Y[1:]
    nbrs = NearestNeighbors(n_neighbors=k).fit(h_x)
    preds = np.zeros((steps, Y.shape[1]))
    curr = Y[-1]
    
    ridge_model = Ridge(alpha=0.1, fit_intercept=False)
    
    for i in range(steps):
        d, idx = nbrs.kneighbors(curr.reshape(1, -1))
        dm = d[0].max() * 1.01
        w = (1 - (d[0]/dm)**3)**3
        W = np.diag(w)
        X_a = np.column_stack((h_x[idx[0]], np.ones(k)))
        
        ridge_model.fit(W @ X_a, W @ h_x1[idx[0]])
        M = ridge_model.coef_.T
        
        curr = np.append(curr, 1) @ M
        preds[i] = curr
    return preds

def evaluate_forecast(t_pred, y_pred, x_ref, y_ref, max_days=30):
    eval_end_time = t_pred[0] + (max_days / 365.25)
    f_end_idx = min(len(y_ref), np.argmin(np.abs(x_ref - eval_end_time)))
    f_start_idx = np.argmin(np.abs(x_ref - t_pred[0]))
    
    if f_start_idx >= f_end_idx or len(y_ref[f_start_idx:f_end_idx]) < 2: 
        return np.nan, np.nan, np.nan
        
    actual_slice = y_ref[f_start_idx:f_end_idx]
    pred_interp = np.interp(x_ref[f_start_idx:f_end_idx], t_pred, y_pred)
    
    rmse = np.sqrt(np.mean((pred_interp - actual_slice)**2))
    mae = np.mean(np.abs(pred_interp - actual_slice))
    corr, _ = pearsonr(pred_interp, actual_slice + np.random.normal(0, 1e-10, len(actual_slice))) 
    
    return rmse, mae, corr

def detect_predicted_event(t_pred, y_pred):
    dy = np.gradient(y_pred)
    # Predict event if velocity drops below 2 standard deviations of the quiet period
    threshold = np.mean(dy) - 2 * np.std(dy) 
    onset_indices = np.where(dy < threshold)[0]
    
    if len(onset_indices) > 0:
        event_idx = onset_indices[0]
    else:
        # Fallback if no threshold is broken
        event_idx = np.argmin(dy)
        
    return t_pred[event_idx], y_pred[event_idx]


if __name__ == '__main__':
    print("\n" + "="*120)
    print("STABILITY ANALYSIS (2016, 2018, 2021, 2024)")
    print("="*120)
    
    # List of specific epochs
    EPOCHS_TO_TEST = ['2016a', '2016b', '2018', '2021', '2024']
    
    longitudinal_results = []
    
    for station in STATIONS:
        x_p, y_p, x_r, y_r, y_s = load_data(station)
        print(f"\n>>> Running Multi-Event Stability Analysis: {station}")
        
        for epoch in EPOCHS_TO_TEST:
            target_date = TARGET_DATES[epoch]
            
            for method in ['model', 'raw', 'savgol']:
                y_active = {'model': y_p, 'raw': y_r, 'savgol': y_s}[method]
                x_active = x_p if method == 'model' else x_r
                cfg = PHASE_CONFIG[station][method]
                dt_active = x_active[1] - x_active[0]
                
                # Calculate base index for the forecast start (17.5 days before target)
                base_idx = np.argmin(np.abs(x_active - (target_date - HORIZON_TEST_DAYS/365.25)))
                
                ensemble_errors = []
                
                # Perform the Jitter Test to calculate Stability Sigma
                jitter_range = int(JITTER_DAYS / (dt_active * 365.25))
                # Generate 5-10 jittered samples across the +/- 2 day window
                sample_step = max(1, (2 * jitter_range) // 8) 
                
                for j in range(-jitter_range, jitter_range + 1, sample_step):
                    idx_j = base_idx + j
                    
                    if idx_j < (cfg['m'] * cfg['tau']) or idx_j >= len(y_active):
                        continue
                        
                    Y_ens = delay_vectors(y_active[:idx_j], cfg['m'], cfg['tau'])
                    k_opt = find_optimal_k(Y_ens)
                    
                    # Run Forecast
                    preds = knn_forecast(Y_ens, int(FORECAST_DURATION_YRS/dt_active), k_opt)[:, 0]
                    t_f = np.arange(1, len(preds)+1)*dt_active + x_active[idx_j-1]
                    
                    # Detect Event
                    pred_ev_time, _ = detect_predicted_event(t_f, preds)
                    err_days = (pred_ev_time - target_date) * 365.25
                    ensemble_errors.append(err_days)
                
                if not ensemble_errors:
                    continue

                # Calculate Metrics
                mean_err = np.mean(ensemble_errors)
                std_err = np.std(ensemble_errors)
                
                # Check for Correct Rejection (CR) vs False Positive (FP)
                model_hor = calculate_mle(delay_vectors(y_p, 3, PHASE_CONFIG[station]['model']['tau']), 
                                          x_p[1]-x_p[0], 3, PHASE_CONFIG[station]['model']['tau'])[1]
                
                formatted_val = f"{mean_err:+.1f}d"
                if 'Quiet' in epoch:
                    if mean_err > model_hor:
                        formatted_val = f"CR (+{mean_err:.1f}d)"
                    else:
                        formatted_val = f"FP ({mean_err:+.1f}d)"

                longitudinal_results.append({
                    'Station': station,
                    'Epoch': epoch,
                    'Method': method.upper(),
                    'Mean_Err': formatted_val,
                    'Sigma': round(std_err, 2)
                })

    # --- FORMATTING ---
    df = pd.DataFrame(longitudinal_results)
    
    pivot_df = df.pivot_table(index=['Station', 'Method'], 
                             columns='Epoch', 
                             values=['Mean_Err', 'Sigma'], 
                             aggfunc='first')
    
    pivot_df = pivot_df.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, ascending=True)
    
    print("\n" + "="*120)
    print("FORECAST STABILITY")
    print("="*120)
    print(pivot_df.to_string())
    print("\n* Mean_Err: CR = Correct Rejection, FP = False Positive.")
