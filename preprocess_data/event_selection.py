import numpy as np
import pandas as pd
import requests
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from datetime import datetime


# event selection for every station


STATIONS = ['GISB', 'KOKO', 'MAHI', 'PAWA', 'CKID', 'CNST', 'PORA']
STATION_COORDS = {
    'GISB': {'lat': -38.6353, 'lon': 177.8860},
    'KOKO': {'lat': -39.0161, 'lon': 177.6678},
    'MAHI': {'lat': -39.1526, 'lon': 177.9070},
    'PAWA': {'lat': -40.0331, 'lon': 176.8639},
    'CKID': {'lat': -39.6579, 'lon': 177.0764},
    'CNST' : {'lat': -38.4880, 'lon': 178.2111},
    'PORA' :  {'lat': -40.2664, 'lon': 176.6352}
}

M_VAL = 5.0
R_VAL = 300 
WINDOW_DAYS = 180  
SALIENCY_THRESHOLD = 2.5

# --- CORE FUNCTIONS ---

def get_events(m, r, lat, lon, start_yr, end_yr):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson", 
        "starttime": f"{start_yr}-01-01", 
        "endtime": f"{end_yr}-12-31",
        "minmagnitude": m, 
        "latitude": lat, 
        "longitude": lon, 
        "maxradiuskm": r
    }
    try:
        resp = requests.get(url, params=params).json()
        return [{
            'date': datetime.fromtimestamp(f['properties']['time']/1000.0).year + 
                 (f['properties']['time']/1000.0 - datetime(datetime.fromtimestamp(f['properties']['time']/1000.0).year,1,1).timestamp()) / 
                 (datetime(datetime.fromtimestamp(f['properties']['time']/1000.0).year+1,1,1).timestamp() - datetime(datetime.fromtimestamp(f['properties']['time']/1000.0).year,1,1).timestamp()),
            'mag': f['properties']['mag']  # Added this back in!
        } for f in resp['features']]
    except Exception as e: 
        print(f"Error fetching USGS data: {e}")
        return []

def is_event_salient(t_obs, u_obs, date, threshold=SALIENCY_THRESHOLD):
    window = 14/365
    pre = u_obs[(t_obs >= date - window) & (t_obs < date)]
    post = u_obs[(t_obs > date) & (t_obs <= date + window)]
    if len(pre) < 3 or len(post) < 3: return False
    jump = np.abs(np.median(post) - np.median(pre))
    return jump > threshold

def fast_objective(p, t_seg, y_target, u_red_f, u_blk_f):

    tau, hr, hb, vr, vb, slope, phi_i = p
    t_rel = t_seg - t_seg[0]
    phi = 1.0 - (1.0 - phi_i) * np.exp(-t_rel / tau)
    
    # Physics-based simulation
    u_mod = (1.0 - phi)*(u_red_f(t_rel + hr) + vr) + phi*(u_blk_f(t_rel + hb) + vb) + (slope * t_rel)
    
    mse = np.mean((u_mod - y_target)**2)
    jump_penalty = np.abs(u_mod[0] - 0.0) 
    ptp_penalty = np.abs(np.ptp(u_mod) - np.ptp(y_target))
    
    return mse + (5.0 * ptp_penalty) + (10.0 * jump_penalty)

def collapse_events(registry, window_days=180):
    window_yr = window_days / 365.25
    sorted_events = sorted(registry.items(), key=lambda x: x[0][0])
    
    if not sorted_events: return []

    clusters = []
    current_cluster = [sorted_events[0]]
    for i in range(1, len(sorted_events)):
        if sorted_events[i][0][0] - current_cluster[0][0][0] <= window_yr:
            current_cluster.append(sorted_events[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_events[i]]
    clusters.append(current_cluster)

    collapsed = []
    for cluster in clusters:
        avg_date = np.mean([c[0][0] for c in cluster])
        max_mag = np.max([c[0][1] for c in cluster])
        
        if max_mag >= 7.0:
            combined_regs = {s: True for s in STATIONS}
        else:
            combined_regs = {s: False for s in STATIONS}
            for c in cluster:
                for s in STATIONS:
                    if c[1][s]: combined_regs[s] = True
        
        collapsed.append(((avg_date, max_mag), combined_regs))
    
    return collapsed

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


u_red = phys_mode('strain_files/epsdot_Gr1.1e-06_GISB_big.mat')
u_blk = phys_mode('strain_files/epsdot_Gr2.5e-06_GISB_small.mat')

event_registry = {}

for station in STATIONS:
    print(f"Processing {station}...")
    try:
        df = pd.read_csv(f'detrended_data/{station}_clean.csv')
        t_obs, u_obs = df['Time'].values, df['Disp_mm'].values
    except:
        continue

    catalog = get_events(M_VAL, R_VAL, STATION_COORDS[station]['lat'], STATION_COORDS[station]['lon'], int(t_obs[0]), int(t_obs[-1]))
    
    for e in catalog:
        date_key = round(e['date'], 4)
        mag_key = e['mag']
        event_id = (date_key, mag_key)
        
        if event_id not in event_registry:
            event_registry[event_id] = {s: False for s in STATIONS}
        
        if is_event_salient(t_obs, u_obs, e['date']):
            event_registry[event_id][station] = True

print("\n" + "="*95)
print(f"{'EVENT MATRIX (180-Day Window)':^95}")
print("="*95)

# Collapse events into epochs
collapsed_events = collapse_events(event_registry, window_days=WINDOW_DAYS)

summary_data = []
for (date, mag), registrations in collapsed_events:
    row = {
        'Date': round(date, 4),
        'Mag': round(mag, 1)
    }
    row.update({s: ("X" if registrations[s] else ".") for s in STATIONS})
    
    hits = sum(registrations.values())
    row['Hits'] = hits
    # Global if hit 3+ stations OR magnitude is significant (>=7.0)
    row['Type'] = "GLOBAL" if (hits >= 3 or mag >= 7.0) else "LOCAL"
    
    if hits > 0:
        summary_data.append(row)

report_df = pd.DataFrame(summary_data)
cols = ['Date', 'Mag'] + STATIONS + ['Hits', 'Type']
print(report_df[cols].to_string(index=False))
print("="*95)