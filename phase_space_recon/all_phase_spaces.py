import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# generates phase spaces for all stations: phys filter, savgol filter, and raw data
# uses params found in phase_space_params.py

# --- CONFIGURATION DATA ---
STATIONS = ['GISB', 'KOKO', 'MAHI', 'PAWA', 'CNST', 'CKID', 'PORA']

CONFIG = {
    'model': {'GISB': 90, 'KOKO': 62, 'MAHI': 74, 'PAWA': 78, 'CNST': 81, 'CKID': 77, 'PORA': 88},
    'raw':   {'GISB': 17, 'KOKO': 41, 'MAHI': 36, 'PAWA': 27, 'CNST': 36, 'CKID': 13, 'PORA': 7},
    'savgol':{'GISB': 73, 'KOKO': 41, 'MAHI': 50, 'PAWA': 42, 'CNST': 52, 'CKID': 50, 'PORA': 53}
}

EMBED_DIM = 3

def load_and_preprocess(station):
    model_path = f'phys_fits/{station}_model.csv'
    gps_path = f'detrended_data/{station}_clean.csv'
    
    model_df = pd.read_csv(model_path)
    xl = model_df.iloc[:, 0].values
    yl = model_df.iloc[:, 1].values

    gps_raw = pd.read_csv(gps_path).values
    x_gps = gps_raw[:, 0]
    y_gps_raw = -gps_raw[:, 1] - gps_raw[0, 1]
    p_fit = np.polyfit(x_gps, y_gps_raw, 1)
    y_gps_detrended = y_gps_raw - np.polyval(p_fit, x_gps)
    
    y_savgol = savgol_filter(y_gps_detrended, window_length=31, polyorder=3)
    
    return xl, yl, x_gps, y_gps_detrended, y_savgol

def delay_vectors(data, m, tau):
    n = len(data)
    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        Y[:, i] = data[(m - 1 - i) * tau : n - i * tau]
    return Y

def apply_pca(Y):
    pca = PCA(n_components=Y.shape[1])
    Y_rot = pca.fit_transform(Y)
    # Align PC1, PC2, and PC3 for consistent orientation
    for i in range(Y_rot.shape[1]):
        if np.corrcoef(Y[:, 0], Y_rot[:, i])[0, 1] < 0:
            Y_rot[:, i] *= -1
    return Y_rot

def plot_heatmap_phase(ax, Y, title):
    points = Y.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    time_colors = np.linspace(0, 1, len(segments))
    
    lc = Line3DCollection(segments, cmap='viridis', array=time_colors, linewidth=1.5, alpha=0.9)
    ax.add_collection3d(lc)
    
    ax.set_xlim(Y[:, 0].min(), Y[:, 0].max())
    ax.set_ylim(Y[:, 1].min(), Y[:, 1].max())
    ax.set_zlim(Y[:, 2].min(), Y[:, 2].max())
    
    ax.view_init(elev=0, azim=0)
    
    ax.set_title(title, fontsize=10)
    
    ax.set_axis_off() 

    return lc

# --- EXECUTION ---
if __name__ == '__main__':
    for station in STATIONS:
        print(f"Processing: {station} (PC2 vs PC3 View)")
        
        try:
            xl, yl, x_gps, y_raw, y_savgol = load_and_preprocess(station)
            
            # PANEL 1: Time Series 
            fig_data, (ax_d1, ax_d2, ax_d3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            fig_data.suptitle(f"Temporal Data: {station}", fontsize=14)
            
            ax_d1.scatter(x_gps, y_raw, s=2, color='black', alpha=0.4, label='Raw GPS')
            
            ax_d2.scatter(x_gps, y_raw, s=1.5, color='black', alpha=0.1, label='Raw GPS')
            ax_d2.plot(x_gps, y_savgol, color='red', lw=1.2, label='Savgol Filter')
            
            ax_d3.scatter(x_gps, y_raw, s=1.5, color='black', alpha=0.1, label='Raw GPS')
            ax_d3.plot(xl, yl, color='blue', lw=1.5, label='Physical Model')
            
            # PANEL 2: Phase Space (PC2 vs PC3)
            fig_phase = plt.figure(figsize=(18, 6))
            fig_phase.suptitle(f"Phase Space Evolution (PC2 vs PC3): {station}", fontsize=14)
            
            # Data Processing & Plotting
            datasets = [
                (yl, CONFIG['model'][station], "Model Fit", 131),
                (y_raw, CONFIG['raw'][station], "Raw Data", 132),
                (y_savgol, CONFIG['savgol'][station], "Savgol Filtered", 133)
            ]
            
            last_lc = None
            for data, tau, label, pos in datasets:
                Y_pca = apply_pca(delay_vectors(data, EMBED_DIM, tau))
                ax = fig_phase.add_subplot(pos, projection='3d')
                last_lc = plot_heatmap_phase(ax, Y_pca, f"{label} (τ={tau})")
            
            cbar_ax = fig_phase.add_axes([0.93, 0.15, 0.01, 0.7])
            fig_phase.colorbar(last_lc, cax=cbar_ax).set_label('Time Progression')

            plt.show()
            
        except Exception as e:
            print(f"Error at {station}: {e}")