import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
from nolitsa import delay, dimension
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# get phase space parameters for phys fit, savgol filter, or raw data


# --- CONFIGURATION ---
STATION = 'PAWA'
MAX_LAG = 300 
MAX_DIM = 10
FNN_THRESHOLD = 0.1  

def embed(data, m, tau):
    """Manual delay-coordinate embedding."""
    n = len(data)
    Y = np.zeros((n - (m - 1) * tau, m))
    for i in range(m):
        Y[:, i] = data[i * tau : n - (m - 1 - i) * tau]
    return Y

def load_data():
    model_df = pd.read_csv(f'new_fits/{STATION}_model.csv')
    #model_df = pd.read_csv(f'detrended_data/{STATION}_clean.csv')
    return model_df.iloc[:, 1].values 

def main():    
    yl = load_data()
    scaler = StandardScaler()
    yl_norm = scaler.fit_transform(yl.reshape(-1, 1)).flatten()

    '''
    window_size = 31 #21, 31
    poly_order = 2    
    yl_smooth = savgol_filter(yl_norm, window_length=window_size, polyorder=poly_order)
    '''
    yl_smooth = yl_norm 

    # 1. Time Lag Analysis
    lags = np.arange(1, MAX_LAG)
    ami = np.array([delay.mi(yl_smooth[:-l], yl_smooth[l:]) for l in lags])
    minima = argrelextrema(ami, np.less)[0]
    tau_ami = lags[minima[0]] if len(minima) > 0 else 100
    
    data_acf = acf(yl_smooth, nlags=MAX_LAG, fft=True)
    selected_tau = tau_ami 

    # 2. Embedding Dimension Analysis
    dims = np.arange(1, MAX_DIM + 1)
    f1, f2, f3 = dimension.fnn(yl_smooth, tau=selected_tau, dim=dims)
    E, Es = dimension.afn(yl_smooth, tau=selected_tau, dim=dims)
    E1 = E[1:] / E[:-1]

    # 3. Reconstruction (Forced m=3)
    m_final = 3 
    Y = embed(yl_smooth, m_final, selected_tau)
    pca = PCA(n_components=3)
    Y_rot = pca.fit_transform(Y)

    # Find dimension where FNN < 10%
    threshold_10_percent = 0.1
    try:
        m_opt = dims[np.where(f1 < threshold_10_percent)[0][0]]
    except IndexError:
        m_opt = "Not reached"

    print(f"--- Analysis for {STATION} ---")
    print(f"Time Lag (tau): {selected_tau}")
    print(f"Found m={m_opt}: FNN% = {f1[m_opt-1]*100:.2f}%")
    print(f"Chosen m={m_final}: FNN% = {f1[m_final-1]*100:.2f}%")

    # --- WINDOW 1: TIME LAG DIAGNOSTICS ---
    plt.figure("Time Lag Selection", figsize=(8, 6))
    plt.plot(lags, ami / np.max(ami), label="AMI (Normalized)", color='blue', lw=1.5)
    plt.plot(np.arange(len(data_acf)), data_acf, label="ACF", color='gray', alpha=0.4)
    plt.axvline(selected_tau, color='red', linestyle='--', label=f'Selected Tau={selected_tau}')
    plt.title(f"{STATION}: Time Delay Analysis")
    plt.xlabel("Lag (Days)")
    plt.ylabel("Correlation / Information")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- WINDOW 2: DIMENSION DIAGNOSTICS ---
    plt.figure("Embedding Dimension Analysis", figsize=(8, 6))
    plt.plot(dims, f1, 'ko-', label='FNN %')
    plt.plot(dims[:-1], E1, 'rs--', label='Cao E1 (Saturation)')
    plt.axhline(FNN_THRESHOLD, color='blue', linestyle=':', label='0.1% Threshold')
    plt.yscale('log')
    plt.title(f"{STATION}: Embedding Dimension Analysis")
    plt.xlabel("Dimension (m)")
    plt.ylabel("FNN Fraction (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # --- WINDOW 3: 3D PHASE SPACE ---
    fig3 = plt.figure("Phase Space Reconstruction", figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    points = Y_rot.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t_gradient = np.linspace(0, 1, len(Y_rot))
    
    lc = Line3DCollection(segments, cmap='viridis', array=t_gradient, linewidths=1.5)
    line = ax3.add_collection3d(lc)

    # Set limits based on PCA data
    ax3.set_xlim(Y_rot[:,0].min(), Y_rot[:,0].max())
    ax3.set_ylim(Y_rot[:,1].min(), Y_rot[:,1].max())
    ax3.set_zlim(Y_rot[:,2].min(), Y_rot[:,2].max())
    
    cbar = fig3.colorbar(line, ax=ax3, fraction=0.02, pad=0.1)
    cbar.set_label('Time Evolution (Normalized)')
    
    ax3.set_title(f"{STATION} Phase Space: m=3, Tau={selected_tau}")
    ax3.set_axis_off()
    ax3.view_init(elev=20, azim=45)

    plt.show()

if __name__ == '__main__':
    main()