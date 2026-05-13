import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr
import warnings

# forecast for specific station and date 
# prints forecasts starting 30 days before event 
# and then mutliple forecasts starting on 17.5 days before to check convergence


warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. USER CONTROL PANEL ---
STATION = 'CKID' 
TARGET_DATE = 2016.9458  #2024.9233 #2021.1886 #2016.6627
ROLLING_WINDOW_DAYS = 30
HORIZON_TEST_DAYS = 17.5 
FORECAST_DURATION_YRS = 0.25   
EVALUATION_WINDOW_DAYS = 30     

# values from phase_space_params.py
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
        # Find the first neighbor index that is sufficiently far away in time
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


# --- EXECUTION & PLOTTING ---

if __name__ == '__main__':
    x_p, y_p, x_r, y_r, y_s = load_data(STATION)
    
    data_map = {'model': y_p, 'savgol': y_s, 'raw': y_r}
    cfg_all = PHASE_CONFIG[STATION]
    
    rolling_acc_data = []
    ensemble_acc_data = []
    mle_data = []
    k_table_ens = pd.DataFrame() 
    
    fig_set1, axes_set1 = plt.subplots(2, 3, figsize=(18, 10))
    fig_set1.suptitle(f"{STATION} SET 1: {ROLLING_WINDOW_DAYS}-Day Rolling Forecasts (Eval Window: {EVALUATION_WINDOW_DAYS} Days)", fontsize=16)
    
    fig_set2, axes_set2 = plt.subplots(2, 3, figsize=(18, 10))
    fig_set2.suptitle(f"{STATION} SET 2: {HORIZON_TEST_DAYS}-Day Ensemble Density Masks", fontsize=16)

    methods = ['model', 'savgol', 'raw']
    print(f"\nTarget Event Date: {TARGET_DATE:.4f}")
    
    cmap_forecast = plt.cm.inferno_r 
    norm_forecast = mcolors.Normalize(vmin=0, vmax=ROLLING_WINDOW_DAYS)
    sm_forecast = plt.cm.ScalarMappable(cmap=cmap_forecast, norm=norm_forecast)
    sm_forecast.set_array([])
    
    for col, method in enumerate(methods):
        cfg = cfg_all[method]
        y_active = data_map[method]
        x_active = x_p if method == 'model' else x_r
        dt_active = x_active[1] - x_active[0]
        
        # MLE call updated to include m and tau
        Y_full = delay_vectors(y_active, cfg['m'], cfg['tau'])
        l_max, t_hor = calculate_mle(Y_full, dt_active, cfg['m'], cfg['tau'])
        mle_data.append({'Method': method, 'MLE (yr^-1)': l_max, 'Horizon (days)': t_hor})
        
        gt_color = 'green' if method == 'model' else ('orange' if method == 'savgol' else 'black')
        
        # ---------------------------------------------------------
        # SET 1: ROLLING FORECASTS
        # ---------------------------------------------------------
        ax_s1_zoom = axes_set1[0, col]; ax_s1_wide = axes_set1[1, col] 
        
        s_idx = np.argmin(np.abs(x_active - (TARGET_DATE - ROLLING_WINDOW_DAYS/365.25)))
        e_idx = np.argmin(np.abs(x_active - TARGET_DATE))
        
        eval_window_end_s1 = x_active[s_idx] + (EVALUATION_WINDOW_DAYS / 365.25)
        for ax in [ax_s1_zoom, ax_s1_wide]:
            ax.axvline(x_active[s_idx], color='gray', linestyle='--', alpha=0.5)

        print(f"\n--- Processing {method.upper()} Rolling Predictions ---")
        for count, i in enumerate(range(s_idx, e_idx + 1)):
            Y_tmp = delay_vectors(y_active[:i], cfg['m'], cfg['tau'])
            k_opt = find_optimal_k(Y_tmp)
            p = knn_forecast(Y_tmp, int(FORECAST_DURATION_YRS/dt_active), k_opt)
            
            t_str = np.concatenate([[x_active[i-1]], np.arange(1, len(p)+1)*dt_active + x_active[i-1]])
            y_str = np.concatenate([[y_active[i-1]], p[:, 0]])
            
            pred_ev_time, pred_ev_y = detect_predicted_event(t_str, y_str)
            time_err_days = (pred_ev_time - TARGET_DATE) * 365.25
            
            days_out = round((TARGET_DATE - x_active[i-1]) * 365.25, 1)
            if count % 3 == 0 or i == e_idx:
                print(f"[{days_out:04.1f} days to target] Event predicted at {pred_ev_time:.4f} | Error: {time_err_days:+.1f} days")

            rmse_raw, mae_raw, corr_raw = evaluate_forecast(t_str, y_str, x_r, y_r, EVALUATION_WINDOW_DAYS)
            rmse_mod, mae_mod, corr_mod = evaluate_forecast(t_str, y_str, x_p, y_p, EVALUATION_WINDOW_DAYS)
            
            rolling_acc_data.append({
                'Method': method,
                'Days_To_Target': days_out,
                'RMSE_vs_Mod': rmse_mod,
                'MAE_vs_Mod': mae_mod,
                'Corr_vs_Mod': corr_mod,
                'Pred_Event_Time': pred_ev_time,
                'Time_Error_Days': time_err_days
            })

            for ax in [ax_s1_zoom, ax_s1_wide]:
                if count == 0: 
                    ax.plot(x_active[:s_idx], y_active[:s_idx], color=gt_color, lw=1.5, alpha=0.8)
                    ax.plot(x_active[s_idx:], y_active[s_idx:], color=gt_color, lw=1.5, alpha=0.15)
                
                ax.plot(t_str, y_str, color=sm_forecast.to_rgba(days_out), lw=0.7, alpha=0.5)

        # ---------------------------------------------------------
        # SET 2: ENSEMBLE DENSITY 
        # ---------------------------------------------------------
        ax_s2_zoom = axes_set2[0, col]; ax_s2_wide = axes_set2[1, col] 
        
        t_ens_start = TARGET_DATE - (HORIZON_TEST_DAYS/365.25)
        idx_e = np.argmin(np.abs(x_active - t_ens_start))
            
        eval_window_end_s2 = x_active[idx_e] + (EVALUATION_WINDOW_DAYS / 365.25)
        for ax in [ax_s2_zoom, ax_s2_wide]:
            ax.axvline(x_active[idx_e], color='gray', linestyle='--', alpha=0.5)

        Y_ens = delay_vectors(y_active[:idx_e], cfg['m'], cfg['tau'])
        k_base = find_optimal_k(Y_ens)
        
        ensemble_runs = []
        ensemble_ks = [] 
        for _ in range(10):
            k_jit = int(k_base * np.random.uniform(0.9, 1.1))
            ensemble_ks.append(k_jit)
            p_e = knn_forecast(Y_ens, int(FORECAST_DURATION_YRS/dt_active), k_jit)[:, 0]
            ensemble_runs.append(np.concatenate([[y_active[idx_e-1]], p_e]))
            
        k_table_ens[method] = ensemble_ks 
        ensemble_runs = np.array(ensemble_runs)
        mean_p = np.mean(ensemble_runs, axis=0)
        t_ens = np.concatenate([[x_active[idx_e-1]], np.arange(1, ensemble_runs.shape[1])*dt_active + x_active[idx_e-1]])
        
        rmse_ens_raw, mae_ens_raw, corr_ens_raw = evaluate_forecast(t_ens, mean_p, x_r, y_r, EVALUATION_WINDOW_DAYS)
        rmse_ens_mod, mae_ens_mod, corr_ens_mod = evaluate_forecast(t_ens, mean_p, x_p, y_p, EVALUATION_WINDOW_DAYS)
        
        ens_ev_time, ens_ev_y = detect_predicted_event(t_ens, mean_p)
        ens_time_err = (ens_ev_time - TARGET_DATE) * 365.25
        
        ensemble_acc_data.append({
            'Method': method, 
            'RMSE_vs_Mod': rmse_ens_mod, 
            'MAE_vs_Mod': mae_ens_mod,
            'Corr_vs_Mod': corr_ens_mod,
            'Pred_Event_Time': ens_ev_time,
            'Time_Error_Days': ens_time_err
        })

        for ax in [ax_s2_zoom, ax_s2_wide]:
            ax.plot(x_active[:idx_e], y_active[:idx_e], color=gt_color, lw=1.5, alpha=0.8)
            ax.plot(x_active[idx_e:], y_active[idx_e:], color=gt_color, lw=1.5, alpha=0.15)
            ax.fill_between(t_ens, np.percentile(ensemble_runs, 5, axis=0), 
                            np.percentile(ensemble_runs, 95, axis=0), color='blue', alpha=0.2)
            ax.plot(t_ens, mean_p, color='red', lw=2)

        # ---------------------------------------------------------
        # FORMATTING
        # ---------------------------------------------------------
        for fig_axes in [axes_set1, axes_set2]:
            ax_z = fig_axes[0, col]; ax_w = fig_axes[1, col]
            for ax in [ax_z, ax_w]:
                ax.scatter(x_r, y_r, s=2, color='black', alpha=0.2) 
                if method == 'raw':
                    ax.plot(x_r, y_r, color='black', lw=0.5, alpha=0.3) 
                ax.set_ylim(-25, 25); ax.grid(alpha=0.2)
            
            ax_z.set_title(f"{method.upper()} - Zoomed (1 Yr)"); ax_z.set_xlim(TARGET_DATE - 1.5, TARGET_DATE + 1.5)
            ax_w.set_title(f"{method.upper()} - Wide (5 Yr)"); ax_w.set_xlim(TARGET_DATE - 2.5, TARGET_DATE + 2.5)

    cax = fig_set1.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig_set1.colorbar(sm_forecast, cax = cax, ax=axes_set1.ravel().tolist(), orientation='vertical', aspect=40, pad=0.02)
    cbar.set_label('Time to Target Event (Days)', fontsize=12, weight='bold')
    cbar.ax.invert_yaxis()

    fig_set1.tight_layout(rect=[0, 0.03, 0.90, 0.95]) 
    fig_set2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 4. PRINTING SUMMARY TABLES ---
    df_roll = pd.DataFrame(rolling_acc_data)
    df_ens = pd.DataFrame(ensemble_acc_data)
    df_mle = pd.DataFrame(mle_data)
    
    print("\n" + "="*110)
    print("GLOBAL PREDICTABILITY (MLE & HORIZON)")
    print("="*110)
    print(df_mle.round(3).to_string(index=False))

    print("\n" + "="*110)
    print(f"SET 1: ROLLING FORECAST (Eval over {EVALUATION_WINDOW_DAYS} days)")
    print(f"Internal Model Noise Floor (Sigma): {cfg_all['model']['sigma']} mm")
    print("="*110)
    
    pivot_roll = df_roll.pivot(index='Days_To_Target', columns='Method', values=['MAE_vs_Mod', 'RMSE_vs_Mod', 'Corr_vs_Mod', 'Time_Error_Days'])
    pivot_roll = pivot_roll.sort_index(ascending=False) 
    print(pivot_roll.round(2).to_string())

    print("\n" + "="*110)
    print(f"SET 2: ENSEMBLE MEAN FORECAST (Started {HORIZON_TEST_DAYS} days out)")
    print("="*110)
    print(df_ens.round(3).to_string(index=False))

    plt.show()