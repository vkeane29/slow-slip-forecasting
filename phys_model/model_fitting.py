import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
from tqdm import tqdm

# runs 15 ensemble optimization fits for station 
# outputs mean fit and 

STATION_NAME = 'PAWA'
FILE_PATH = f'detrended_data/{STATION_NAME}_clean.csv'
NUM_TRAJECTORIES = 15 

# from clean_data.py
noise_map = {'GISB': 7.64, 'KOKO': 5.3, 'MAHI': 9.08, 'PAWA': 9.13, 'CKID': 5.09, 'CNST': 8.3, 'PORA': 7.78}
NOISE_FLOOR = noise_map.get(STATION_NAME) 

# from event_selection.py
event_map = {
    'GISB': [2003.0271, 2006.3340, 2007.8977, 2015.0049, 2016.6627, 2016.9458, 2021.1886, 2023.3942, 2024.9233],
    
    'KOKO': [2006.3340, 2007.8977, 2010.2053, 2011.9700, 2013.4926, 2015.0049, 2016.6627, 2016.9458, 2018.9955, 2021.1886, 2022.9878, 2023.3942, 2024.9233],
    
    'MAHI': [2007.8977, 2009.1452, 2010.2053, 2011.4161, 2011.9700, 2013.4926, 2015.0049, 2016.6627, 2016.9458, 2018.9955, 2019.5520, 2021.1886, 2022.9878, 2023.3942, 2024.9233],
    
    'PAWA': [2006.3340, 2006.9624, 2010.2053, 2012.6343, 2016.6627, 2016.9458, 2021.1886, 2022.9878, 2023.3942, 2024.9233],
    
    'CKID': [2006.9624, 2007.8977, 2010.2053, 2012.6343, 2015.0049, 2016.6627, 2016.9458, 2021.1886, 2022.9878],
    
    'CNST': [2007.8977, 2008.5873, 2010.2053, 2015.0049, 2016.6627, 2016.9458, 2017.8247, 2018.9955, 2020.0354, 2021.1886, 2023.3942, 2024.9233],
    
    'PORA': [2006.9624, 2007.8977, 2009.5666, 2011.4161, 2012.6343, 2016.6627, 2016.9458, 2021.1886]
}



EVENT_DATES = event_map.get(STATION_NAME)

def phys_mode(path):
    data = sio.loadmat(path)
    eps_dot, times = data['eps_dot'], data['times'].flatten()
    Kappa_m, half_width, gamma0 = 0.9e-6, 3.15, 3.3e5
    gamma_dot_zero = Kappa_m * gamma0 / (half_width**2)
    x = np.cos(np.arange(eps_dot.shape[1]) * np.pi / (eps_dot.shape[1] - 1))
    V = np.ones_like(eps_dot) * (2 * half_width * gamma_dot_zero)
    for i in range(1, eps_dot.shape[1]):
        V[:, i] = V[:, i-1] + 2 * half_width * gamma_dot_zero * \
                  (np.maximum(0, eps_dot[:, i-1]) + np.maximum(0, eps_dot[:, i])) * \
                  (x[i-1] - x[i]) / 2
    V = V - V[:, -1:]
    dt = np.diff(times, prepend=times[0])
    u = np.zeros_like(V)
    for k in range(1, len(times)):
        u[k, :] = u[k-1, :] + (half_width**2 / Kappa_m) * ((V[k-1, :] + V[k, :]) * dt[k] / 2)
    time_yr = (half_width**2 / Kappa_m * times) / 3.15569e7
    return interp1d(time_yr, u[:, 0] * 2000.0, fill_value="extrapolate")

def simulate_modes(t_seg, u_red_f, u_blk_f, params):
    tau, hr, hb, vr, vb, slope, phi_i = params
    t_rel = t_seg - t_seg[0] 
    phi = 1.0 - (1.0 - phi_i) * np.exp(-t_rel / tau)
    u_phys = (1.0 - phi)*(u_red_f(t_rel + hr) + vr) + phi*(u_blk_f(t_rel + hb) + vb)
    u_model = u_phys + (slope * t_rel)
    return u_model, phi

def objective(p, t_seg, y_target, u_red_f, u_blk_f, last_u_end):
    u_sim, _ = simulate_modes(t_seg, u_red_f, u_blk_f, p)
    mse = np.mean((u_sim - y_target)**2)
    jump_penalty = np.abs(u_sim[0] - 0.0) # Forcing alignment to the relative 0
    ptp_penalty = np.abs(np.ptp(u_sim) - np.ptp(y_target))
    return mse + (5.0 * ptp_penalty) + (10.0 * jump_penalty)

u_red_func = phys_mode('strain_files/epsdot_Gr1.1e-06_GISB_big.mat')
u_blk_func = phys_mode('strain_files/epsdot_Gr2.5e-06_GISB_small.mat')

gps = pd.read_csv(FILE_PATH)
t_obs = gps.iloc[:, 0].values 
u_obs = -gps.iloc[:, 1].values 
u_obs -= u_obs[0]

p_fit = np.polyfit(t_obs, u_obs, 1)
u_obs = u_obs - np.polyval(p_fit, t_obs)

boundaries = sorted(list(set([t_obs[0]] + EVENT_DATES + [t_obs[-1]])))

# --- ENSEMBLE EXECUTION ---
all_trajectories = []
final_time_axis = []

# optimization bounds
bounds = [(0.1, 3.0),  #tau
          (-5.0, 5.0), #horiz r
          (-5.0, 5.0), #horiz b
          (-8.0, 8.0), #vert r
          (-8.0, 8.0), #vert b
          (15, 25), #slope
          (0.0, 0.25)] #phi_init

print(f"Running {NUM_TRAJECTORIES} ensemble trajectories for {STATION_NAME}...")

num_segments = len(boundaries) - 1
all_params = np.zeros((NUM_TRAJECTORIES, num_segments, 7))

for traj_idx in tqdm(range(NUM_TRAJECTORIES), desc="Ensemble"):
    current_v_offset = u_obs[0]
    full_u_path = []
    
    for i in range(len(boundaries) - 1):
        t_start, t_end = boundaries[i], boundaries[i+1]
        mask = (t_obs >= t_start) & (t_obs < t_end)
        if mask.sum() < 2: continue
        t_seg, y_target = t_obs[mask], u_obs[mask]
        
        # Optimize relative to current_v_offset
        res = differential_evolution(objective, bounds, 
                                     args=(t_seg, y_target - current_v_offset, u_red_func, u_blk_func, 0.0),
                                     popsize=15, tol=0.01)
        

        # Store the 7 parameters for this trajectory and segment
        all_params[traj_idx, i, :] = res.x

        u_fit, _ = simulate_modes(t_seg, u_red_func, u_blk_func, res.x)
        u_continuous = u_fit + current_v_offset
        
        full_u_path.extend(u_continuous)
        if traj_idx == 0:
            final_time_axis.extend(t_seg)
        
        current_v_offset = u_continuous[-1]
        
    all_trajectories.append(full_u_path)

# SUMMARY STATISTICS FOR PARAMS
param_names = ['tau', 'hr', 'hb', 'vr', 'vb', 'slope', 'phi_i']
summary_data = []

for i in range(num_segments):
    t_start, t_end = boundaries[i], boundaries[i+1]
    seg_label = f"{t_start:.2f}-{t_end:.2f}"
    
    row = {'Segment': seg_label}
    for p_idx, p_name in enumerate(param_names):
        p_mean = np.mean(all_params[:, i, p_idx])
        p_std = np.std(all_params[:, i, p_idx])
        row[p_name] = f"{p_mean:.2f}±{p_std:.2f}"
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)

print("\n" + "="*110)
print(f" ENSEMBLE PARAMETER STATS (Mean ± 1σ) for {STATION_NAME}")
print("-" * 110)
print(df_summary.to_string(index=False))
print("="*110)

# --- PARAMETER EVOLUTION PLOT WITH UNCERTAINTY BARS ---
param_labels = [r'$\tau$ (Time Const)', '$h_r$ (Red Shift)', '$h_b$ (Black Shift)', 
                '$v_r$ (Red Offset)', '$v_b$ (Black Offset)', 'Slope (mm/yr)', '$\phi_i$ (Weighting)']

# Calculate segment midpoints for the x-axis
seg_centers = [(boundaries[i] + boundaries[i+1])/2 for i in range(len(boundaries)-1)]

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
axes = axes.flatten()

for p_idx in range(7):
    # Extract mean and std for the specific parameter across all segments
    means = np.mean(all_params[:, :, p_idx], axis=0)
    stds = np.std(all_params[:, :, p_idx], axis=0)
    
    # Plotting error bars
    axes[p_idx].errorbar(seg_centers, means, yerr=stds, fmt='o-', 
                         color='darkblue', ecolor='red', capsize=4, 
                         markersize=6, elinewidth=1.5, alpha=0.8)
    
    # Add horizontal lines for the bounds to see "clipping"
    axes[p_idx].axhline(bounds[p_idx][0], color='black', ls='--', alpha=0.2)
    axes[p_idx].axhline(bounds[p_idx][1], color='black', ls='--', alpha=0.2)
    
    axes[p_idx].set_title(param_labels[p_idx], fontsize=12, fontweight='bold')
    axes[p_idx].grid(True, alpha=0.2)
    
    if p_idx >= 5: # Only label the bottom x-axes
        axes[p_idx].set_xlabel("Year")

fig.delaxes(axes[-1])

plt.suptitle(f"Parameter Stability & Uncertainty: Station {STATION_NAME}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- ENSEMBLE CALCULATIONS ---
all_trajectories = np.array(all_trajectories)
mean_u = np.mean(all_trajectories, axis=0)
std_u = np.std(all_trajectories, axis=0)

# --- ACCURACY METRICS ---
mask_sync = np.isin(t_obs, final_time_axis)
y_true = u_obs[mask_sync]
y_pred = mean_u
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
r2 = r2_score(y_true, y_pred)

# --- PLOTTING ---
plt.figure(figsize=(14, 7))

# 1. Raw GPS Data 
plt.scatter(t_obs, u_obs, s=1.5, color='black', alpha=0.3, label='GPS Data', zorder=1)

# 2. Density mask
plt.fill_between(final_time_axis, mean_u - 2*std_u, mean_u + 2*std_u, 
                 color='crimson', alpha=0.15, label='95% Confidence ($2\sigma$)', zorder=2)

# 3. ensemble paths
for i in range(all_trajectories.shape[0]):
    plt.plot(final_time_axis, all_trajectories[i], color='lightgray', alpha=0.05, lw=0.5, zorder=3)

# 4. mean fit
plt.plot(final_time_axis, mean_u, color='red', lw=2.5, label='Ensemble Mean Fit', zorder=4)

# event markers
for date in EVENT_DATES:
    plt.axvline(date, color='blue', ls=':', alpha=0.3, lw=1)

plt.ylabel("Displacement (mm)")
plt.xlabel("Year")
plt.grid(alpha=0.1)
plt.tight_layout()
plt.show()

# --- FINAL REPORT ---
avg_model_err = np.mean(std_u)
combined_uncertainty = np.sqrt(avg_model_err**2 + NOISE_FLOOR**2)

print("\n" + "="*55)
print(f" PHYSICS-BASED FIT REPORT: {STATION_NAME}")
print("-" * 55)
print(f" ACCURACY SCORE (R^2):    {r2:.4f}")
print(f" RMSE:                    {rmse:.3f} mm")
print(f" MODEL DISAGREEMENT (σ):  {avg_model_err:.3f} mm")
print(f" TOTAL UNCERTAINTY:       {combined_uncertainty:.3f} mm")
print("-" * 55)
print(f" EXPORTED TO: phys_fits/{STATION_NAME}_model.csv")
print("="*55)

# Export for Phase Space script
export_df = pd.DataFrame({'Time': final_time_axis, 'Mean_Model_mm': mean_u, 'Std_Dev': std_u})
export_df.to_csv(f'new_fits/{STATION_NAME}_model.csv', index=False)