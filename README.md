library imports
pandas, numpy, matplotlib, scipy, sklearn, nolitsa, mpl_toolkits, 
requests, io, tqdm, datetime, statsmodels, warnings



MODEL PIPELINE

data_downloading
  run load_data.py 
  - with desired station codes
    
  run get_timeser.py
  - saves separate station time series to timeser folder


preprocess_data
  run clean_data.py
  - evaluates noise of each station, detrends, and resaves cleaned data

  run event_select.py
  - gets event dates per station


phys_model
  run model_clipping (uses strain_files)
  - saves averaged model fit from 15 ensemble run


phase_space_recon
  run phase_space_params.py 
  - get phys mode, savgol model, and raw data tau/dim values
  
  run all_phase_spaces.py
  - constructs phase space from phys mode, savgol model, and raw data


forecasting
  rolling_forecast.py
  - forecasts for specific station and event time (phys filter, savgol, and raw data)
  - figures
     - forecasts starting 30 days before event
     - multiple forecasts starting 17.5 days before event

  stability_forecasts.py
  - runs forecasts on all stations for multiple times (phys filter, savgol, and raw data)
  - checks for event prediction correct rejection/false positive
  - calculates convergence of forecasts

  MLE_analysis.py
  - calculates MLE for phys filter, savgol, and raw data
  - runs calc over multiple years to check convergence/stability
    
