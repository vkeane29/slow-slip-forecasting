import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. USER CONTROL PANEL ---
STATION = 'PAWA' 
TARGET_DATE = 2024.9233
ROLLING_WINDOW_DAYS = 100
FORECAST_DURATION_YRS = 1.0 
EMBED_DIM = 3

PHASE_CONFIG = {
    'GISB': {'model': {'tau': 90, 'm': 3, 'sigma': 7.87}, 'raw': {'tau': 17, 'm': 3, 'sigma': 7.64}, 'savgol': {'tau': 73, 'm': 3, 'sigma': 7.64}},
    'KOKO': {'model': {'tau': 62, 'm': 3, 'sigma': 5.57}, 'raw': {'tau': 41, 'm': 3, 'sigma': 5.30}, 'savgol': {'tau': 41, 'm': 3, 'sigma': 5.30}},
    'MAHI': {'model': {'tau': 74, 'm': 3, 'sigma': 9.16}, 'raw': {'tau': 36, 'm': 3, 'sigma': 9.08}, 'savgol': {'tau': 50, 'm': 3, 'sigma': 9.08}},
    'PAWA': {'model': {'tau': 91, 'm': 3, 'sigma': 9.55}, 'raw': {'tau': 27, 'm': 3, 'sigma': 9.13}, 'savgol': {'tau': 42, 'm': 3, 'sigma': 9.13}},
    'CKID': {'model': {'tau': 77, 'm': 3, 'sigma': 5.62}, 'raw': {'tau': 13, 'm': 3, 'sigma': 5.09}, 'savgol': {'tau': 50, 'm': 3, 'sigma': 5.09}},
    'CNST': {'model': {'tau': 81, 'm': 3, 'sigma': 8.50}, 'raw': {'tau': 36, 'm': 3, 'sigma': 8.30}, 'savgol': {'tau': 52, 'm': 3, 'sigma': 8.30}},
    'PORA': {'model': {'tau': 88, 'm': 3, 'sigma': 8.30}, 'raw': {'tau': 7,  'm': 3, 'sigma': 7.78}, 'savgol': {'tau': 53, 'm': 3, 'sigma': 7.78}}
}

# --- 2. CORE FUNCTIONS ---

def load_data(station):
    model_df = pd.read_csv(f'new_fits/{station}_model.csv')
    raw_df = pd.read_csv(f'detrended_data/{station}_clean.csv')
    x_p, y_p = model_df.iloc[:, 0].values, model_df.iloc[:, 1].values
    phi_p = model_df['Mean_Phi'].values if 'Mean_Phi' in model_df.columns else np.zeros_like(x_p)
    
    x_r = raw_df.iloc[:, 0].values
    y_r = -raw_df.iloc[:, 1].values - raw_df.iloc[0, 1]
    p_fit = np.polyfit(x_r, y_r, 1)
    y_r = y_r - np.polyval(p_fit, x_r)
    y_s = savgol_filter(y_r, window_length=31, polyorder=2)
    return x_p, y_p, phi_p, x_r, y_r, y_s

def delay_vectors(data, m, tau):
    m, tau = int(m), int(tau)
    n = len(data)
    Y = np.zeros((int(n - (m - 1) * tau), m))
    for i in range(m):
        Y[:, i] = data[(m - 1 - i) * tau : n - i * tau]
    return Y

def apply_pca_with_transformer(Y):
    pca = PCA(n_components=Y.shape[1])
    Y_rot = pca.fit_transform(Y)
    
    sign_vector = np.ones(Y_rot.shape[1])
    for i in range(Y_rot.shape[1]):
        if np.corrcoef(Y[:, 0], Y_rot[:, i])[0, 1] < 0:
            sign_vector[i] = -1.0
            Y_rot[:, i] *= -1.0
    return Y_rot, pca, sign_vector

def find_optimal_k(Y):
    K_RANGE = np.arange(60, 200, 10)
    val_size = 10
    if len(Y) < val_size + max(K_RANGE) + 2:
        return 60  
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
            if dm == 0: dm = 1e-6 
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

def forecast_triggered_phi(t_forecast, y_forecast, phi_initial, tau_days):
    tau_yrs = tau_days / 365.25
    phi_preds = np.zeros_like(t_forecast)
    dy_dt = np.gradient(y_forecast, t_forecast)
    trigger_threshold = np.mean(dy_dt) - 2.0 * np.std(dy_dt)
    
    event_triggered = False
    t_event_onset = 0.0
    
    for idx in range(len(t_forecast)):
        t_curr = t_forecast[idx]
        t_rel_quiet = t_curr - t_forecast[0]
        
        if not event_triggered and dy_dt[idx] < trigger_threshold and idx > 2:
            event_triggered = True
            t_event_onset = t_curr
            
        if not event_triggered:
            phi_preds[idx] = 1.0 - (1.0 - phi_initial) * np.exp(-t_rel_quiet / tau_yrs)
        else:
            t_rel_event = t_curr - t_event_onset
            phi_event_base = 0.05 
            phi_preds[idx] = 1.0 - (1.0 - phi_event_base) * np.exp(-t_rel_event / tau_yrs)
            
    return phi_preds, t_event_onset if event_triggered else None

from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_heatmap_phase_modes(ax, Y, phi_mode, tau, m, title):
    points = Y.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    mode_truncated = phi_mode[(m - 1) * tau:]
    segment_modes = mode_truncated[:-1]
    
    colors_list = []
    for val in segment_modes:
        if val <= 0.5:
            norm_val = val / 0.5
            colors_list.append((0.5 + 0.5 * norm_val, 0.0, 0.0)) 
        else:
            norm_val = (val - 0.5) / 0.5
            gray = 0.4 * (1.0 - norm_val) 
            colors_list.append((gray, gray, gray))
            
    lc = Line3DCollection(segments, colors=colors_list, linewidth=1.5, alpha=0.9)
    ax.add_collection3d(lc)
    
    ax.set_xlim(Y[:, 0].min(), Y[:, 0].max())
    ax.set_ylim(Y[:, 1].min(), Y[:, 1].max())
    ax.set_zlim(Y[:, 2].min(), Y[:, 2].max())
    
    ax.view_init(elev=0, azim=0)
    ax.set_title(title, fontsize=12, weight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.set_zlabel("PC 3", fontsize=9)
    return lc

# --- 3. EXECUTION ---
if __name__ == '__main__':
    x_p, y_p, phi_p, x_r, y_r, y_s = load_data(STATION)
    cfg = PHASE_CONFIG[STATION]['model']
    dt = x_p[1] - x_p[0]
    
    # Storage for error calculations
    forecast_errors = []
    
    cmap_forecast = plt.cm.inferno_r 
    norm_forecast = mcolors.Normalize(vmin=0, vmax=ROLLING_WINDOW_DAYS)
    sm_forecast = plt.cm.ScalarMappable(cmap=cmap_forecast, norm=norm_forecast)
    sm_forecast.set_array([])
    
    fig, (ax_disp, ax_phi) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, 
                                          gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.0})
    fig.suptitle(f"{STATION}: Displacement-Triggered Physical Phase Modeling", fontsize=16, weight='bold')
    
    ax_phi.yaxis.tick_right()
    ax_phi.yaxis.set_label_position("right")
    ax_disp.spines['bottom'].set_visible(False)
    ax_phi.spines['top'].set_visible(False)
    ax_disp.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    fig_phase = plt.figure(figsize=(10, 8))
    ax_phase_3d = fig_phase.add_subplot(111, projection='3d')
    
    s_idx = np.argmin(np.abs(x_p - (TARGET_DATE - ROLLING_WINDOW_DAYS/365.25)))
    e_idx = np.argmin(np.abs(x_p - TARGET_DATE))
    
    ax_disp.axvline(TARGET_DATE, color='purple', linestyle='-', lw=2, zorder=5)
    ax_phi.axvline(TARGET_DATE, color='purple', linestyle='-', lw=2, zorder=5)
    ax_disp.axvline(x_p[s_idx], color='gray', linestyle='--', alpha=0.5)
    ax_phi.axvline(x_p[s_idx], color='gray', linestyle='--', alpha=0.5)

    Y_model_full = delay_vectors(y_p, EMBED_DIM, cfg['tau'])
    Y_model_rot, model_pca_transformer, locked_sign_vector = apply_pca_with_transformer(Y_model_full)
    plot_heatmap_phase_modes(ax_phase_3d, Y_model_rot, phi_p, cfg['tau'], EMBED_DIM, f"Model Attractor Trajectories: {STATION}")

    print(f"--- Processing Trigger-Linked Rolling Tracks for {STATION} ---")
    for count, i in enumerate(range(s_idx, e_idx + 1)):
        Y_tmp = delay_vectors(y_p[:i], cfg['m'], cfg['tau'])
        k_opt = find_optimal_k(Y_tmp)
        p = knn_forecast(Y_tmp, int(FORECAST_DURATION_YRS / dt), k_opt)
        
        t_str = np.concatenate([[x_p[i-1]], np.arange(1, len(p)+1)*dt + x_p[i-1]])
        y_str = np.concatenate([[y_p[i-1]], p[:, 0]])
        
        current_phi_base = phi_p[i-1]
        phi_preds, t_onset = forecast_triggered_phi(t_str, y_str, current_phi_base, tau_days=cfg['tau'])
        
        # Calculate Error
        if t_onset is not None:
            days_off = (t_onset - TARGET_DATE) * 365.25
            days_until = (TARGET_DATE - x_p[i-1]) * 365.25
            forecast_errors.append((days_until, days_off))
        
        days_out = round((TARGET_DATE - x_p[i-1]) * 365.25, 1)
        line_color = sm_forecast.to_rgba(days_out)
        
        if count == 0:
            ax_disp.plot(x_p[:s_idx], y_p[:s_idx], color='red', lw=2.0, alpha=0.8)
        ax_disp.plot(t_str, y_str, color=line_color, lw=1.0, alpha=0.4)
        
        if count == 0:
            ax_phi.plot(x_p[:s_idx], phi_p[:s_idx], color='darkmagenta', lw=2.0, alpha=0.8)
        ax_phi.plot(t_str, phi_preds, color=line_color, lw=1.0, alpha=0.4)
        
        if i > (EMBED_DIM * cfg['tau']):
            proj_steps = min(len(p), len(Y_tmp))
            preds_pca_match = model_pca_transformer.transform(p[:proj_steps])
            preds_pca_match = preds_pca_match * locked_sign_vector
                    
            ax_phase_3d.plot(preds_pca_match[:, 0], preds_pca_match[:, 1], preds_pca_match[:, 2], 
                          color=line_color, lw=1.0, alpha=1.0, zorder=10)
            
            if count == 0:
                ax_phase_3d.scatter(preds_pca_match[0, 0], preds_pca_match[0, 1], preds_pca_match[0, 2],
                                 color='red', marker='x', s=75, lw=2.0, zorder=15)

    ax_disp.scatter(x_r, y_r, s=20.0, facecolor='white', edgecolor='black', alpha=0.15, zorder=1)
    ax_disp.set_ylabel("Displacement (mm)", weight='bold')
    ax_disp.set_ylim(-25, 25)
    ax_disp.grid(alpha=0.2)
    ax_disp.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_phi.set_ylabel(r"Structural Mix ($\phi$)", weight='bold', color='darkmagenta')
    ax_phi.set_ylim(-0.05, 1.05)
    ax_phi.grid(alpha=0.2)
    ax_phi.set_xlabel("Time(Year)", weight='bold')
    ax_phi.set_xlim(TARGET_DATE - 6, TARGET_DATE + 2.0)
    
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm_forecast, cax=cax, orientation='vertical')
    cbar.set_label('Days until event', fontsize=11, weight='bold')
    cbar.ax.invert_yaxis()
    fig.subplots_adjust(left=0.06, right=0.86, top=0.90, bottom=0.12)

    # --- FIGURE 3: FORECAST ACCURACY ---
    if forecast_errors:
        errors = np.array([x[1] for x in forecast_errors])
        print(f"\n--- Statistics ---")
        print(f"Average Forecast Error: {np.mean(errors):.2f} +/- {np.std(errors):.2f} days")
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        x_plot = [x[0] for x in forecast_errors]
        y_plot = [x[1] for x in forecast_errors]
        
        ax3.scatter(x_plot, y_plot, alpha=0.6, c='purple')
        ax3.axhline(0, color='black', linestyle='--')
        ax3.set_xlabel("Days until event (100 -> 0)", weight='bold')
        ax3.set_ylabel("Days off actual event", weight='bold')
        ax3.set_title(f"Forecast Accuracy vs Lead Time: {STATION}", weight='bold')
        ax3.set_xlim(100, 0)
        ax3.grid(True, alpha=0.3)

    plt.show()
