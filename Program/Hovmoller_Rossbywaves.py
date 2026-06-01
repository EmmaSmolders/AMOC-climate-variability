#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:13:41 2025

@author: 6008399

Hovmoller diagrams of zonal subsurface temperature (meridionally averaged in latitude band)

"""

#%%

from pylab import *
import numpy
import datetime
import time
import math
import netCDF4 as netcdf
import matplotlib.colors as colors
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pylab import *
import numpy
import datetime
import time
import glob, os
import math
import netCDF4 as netcdf
import matplotlib.colors as colors
from scipy import stats
from cartopy import crs as ccrs, feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
import statsmodels.api as sm
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
import cartopy.mpl.ticker as cticker
import numpy as np
import numpy.ma as ma
from scipy.signal import butter, filtfilt
#import pywt 

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

# Function to apply band-pass filter (15-30 year component)
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

# Function to apply Morlet wavelet transform and extract a specific frequency band
def morlet_wavelet_filter(data, dt, low_period, high_period, wavelet='cmor'):
    """
    Apply Morlet wavelet transform and extract a specific frequency band.
    
    Parameters:
        data: 1D array, the time series to filter.
        dt: float, time step of the data.
        low_period: float, lower bound of the period (e.g., 15 years).
        high_period: float, upper bound of the period (e.g., 30 years).
        wavelet: str, type of wavelet (default is 'cmor' for Morlet).
    
    Returns:
        filtered_data: 1D array, the reconstructed time series in the specified period band.
    """
    # Perform continuous wavelet transform (CWT)
    scales = np.arange(1, 500)  # Define scales (adjust as needed)
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, dt)
    
    # Convert periods to scales
    periods = 1 / frequencies
    low_scale = np.argmin(np.abs(periods - low_period))
    high_scale = np.argmin(np.abs(periods - high_period))
    
    # Extract the desired scales (frequency band)
    band_coefficients = coefficients[low_scale:high_scale + 1, :]
    
    # Reconstruct the filtered signal by summing over the selected scales
    filtered_data = np.sum(band_coefficients, axis=0)
    
    return filtered_data

def SST_pointwise_detrending(sst, time, degree=2, mask=None):
    """
    Perform pointwise polynomial detrending (linear or quadratic) on SST data.

    Parameters
    ----------
    sst : np.ndarray
        3D array [time, lat, lon] of SST values
    time : np.ndarray
        1D array of time values (numeric, same length as sst.shape[0])
    degree : int, optional
        Polynomial degree (1=linear, 2=quadratic)
    mask : np.ndarray, optional
        2D boolean or 0/1 mask [lat, lon] (True or 1 = ocean)

    Returns
    -------
    sst_detrended : np.ndarray
        Detrended SST field, same shape as input
    """

    assert sst.ndim == 3, "sst must be [time, lat, lon]"
    assert time.ndim == 1, "time must be 1D"
    assert len(time) == sst.shape[0], "time dimension mismatch"
    assert degree in [1, 2], "degree must be 1 or 2"

    Nt, Ny, Nx = sst.shape
    sst_flat = sst.reshape(Nt, Ny * Nx)  # shape (Nt, Ngrid)

    # Apply mask if given
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
    else:
        mask_flat = np.ones(Ny * Nx, dtype=bool)

    sst_detrended = np.full_like(sst_flat, np.nan)

    # Construct design matrix for polyfit
    X = np.vander(time, degree + 1)  # e.g. [t^2, t^1, 1] if degree=2

    # Precompute pseudoinverse (for speed)
    pinv_X = np.linalg.pinv(X)

    # Fit polynomial at each grid cell
    valid_idx = np.where(mask_flat)[0]

    for idx in valid_idx:
        y = sst_flat[:, idx]
        if np.all(np.isnan(y)):
            continue

        # Fit polynomial coefficients
        coeffs = pinv_X.dot(y)  # shape (degree+1,)

        # Compute fitted trend
        trend = X.dot(coeffs)

        # Subtract trend
        sst_detrended[:, idx] = y - trend

    # Reshape back to [time, lat, lon]
    sst_detrended = sst_detrended.reshape((Nt, Ny, Nx))

    return sst_detrended


#def running_mean(x, N):
#    return np.convolve(x, np.ones(N)/N, mode="valid")

def TrendRemover(time, data, trend_type):
	"""Removes trend of choice"""
	
	rank = polyfit(time, data, trend_type)
	fitting = 0.0 
		
	for rank_i in range(len(rank)):
			
		fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))

	data -= fitting
	
	return data

import numpy as np
import numpy.ma as ma
from scipy.signal import butter, filtfilt

def running_mean(x, w):
    # centered running mean; edges become nan
    x = np.asarray(x, dtype=float)
    kernel = np.ones(w) / w
    y = np.convolve(x, kernel, mode="same")
    # mark edges as nan where the window is incomplete
    half = w // 2
    y[:half] = np.nan
    y[-half:] = np.nan
    return y

def remove_lowfreq_component(X, w=101):
    """
    X: (time, lon) array, can be masked.
    Removes time-mean and subtracts a w-year running mean per lon.
    """
    Xm = ma.array(X, copy=True)

    # remove time mean per lon (anomalies)
    Xm = Xm - ma.mean(Xm, axis=0)

    out = ma.masked_all(Xm.shape)
    for j in range(Xm.shape[1]):
        x = Xm[:, j]
        if ma.is_masked(x):
            # fill masked with nan for smoothing
            xf = x.filled(np.nan)
        else:
            xf = np.asarray(x, dtype=float)

        lp = running_mean(xf, w=w)      # low-pass estimate
        hp = xf - lp                    # high-pass residual
        out[:, j] = ma.masked_invalid(hp)

    return out

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # filtfilt cannot handle NaNs well; filter each lon separately
    X = ma.array(data, copy=True)
    Y = ma.masked_all(X.shape)

    for j in range(X.shape[1]):
        x = X[:, j].filled(np.nan)
        good = np.isfinite(x)
        if good.sum() < 10 * order:
            continue

        # filter only contiguous valid segment(s) - simplest: require all-valid
        # (If you have NaNs, you can interpolate or segment; ideally your ocean mask yields full series per bin.)
        if not np.all(good):
            # quick linear interp over NaNs (okay if few)
            t = np.arange(len(x))
            x[~good] = np.interp(t[~good], t[good], x[good])

        y = filtfilt(b, a, x)
        Y[:, j] = y

    return Y


#%% Read in data

#Choose which months you want (i.e. DJF or JJA)
month_start = 1
month_end   = 12

ts = 'yearly' #or 'montly' or 'yearly'

#Note that the 100-300m depth averaged temperature has a different lon-lat grid than the 100-1100m depth averaged temperature
#fh      = netcdf.Dataset(directory_data+'SST_month_'+str(month_start)+'-'+str(month_end)+'_QE_year_0-2200.nc', 'r')

fh       = netcdf.Dataset(directory_data + 'TEMP_Atlantic_depth_averaged_100_300m_year_600-1500_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_forward      = fh.variables['time_month'][:]                     #Model years
    SST_forward       = fh.variables['TEMP_month'][:,220::,80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_forward      = fh.variables['time'][:]                     #Model years
    SST_forward       = fh.variables['TEMP'][:,220::,80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
    
lon               = fh.variables['lon'][220::,80:80+100]            #Array of longitudes [degE]
lat               = fh.variables['lat'][220::,80:80+100]            #Array of latitudes [degN]
area              = fh.variables['area'][220::,80:80+100]


fh.close()

plt.figure()
plt.contourf(lon, lat, SST_forward[0])

fh       = netcdf.Dataset(directory_data + 'TEMP_Atlantic_depth_averaged_100_300m_year_2900-3800_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_backward      = fh.variables['time_month'][:]                     #Model years
    SST_backward       = fh.variables['TEMP_month'][:,220::,80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_backward      = fh.variables['time'][:]                     #Model years
    SST_backward       = fh.variables['TEMP'][:,220::,80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
fh.close()

plt.figure()
plt.contourf(lon, lat, SST_backward[0])

fh       = netcdf.Dataset(directory_data + 'TEMP_SALT_DENS_Atlantic_depth_averaged_100_1100m_year_600-1500_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_forward      = fh.variables['time_month'][:]                     #Model years
    TEMP_forward_1100       = fh.variables['TEMP_month'][:,220::,0:100]         #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_forward      = fh.variables['time'][:]                     #Model years
    TEMP_forward_1100       = fh.variables['TEMP'][:,220::,0:100]            #Sea level pressure (av\eraged over months) [hPa]
    
lon_temp               = fh.variables['lon'][220::,0:100]            #Array of longitudes [degE]
lat_temp               = fh.variables['lat'][220::,0:100]            #Array of latitudes [degN]
area_temp              = fh.variables['area'][220::,0:100]


fh.close()

plt.figure()
plt.contourf(lon_temp, lat_temp, TEMP_forward_1100[0])

fh       = netcdf.Dataset(directory_data + 'TEMP_SALT_DENS_Atlantic_depth_averaged_100_1100m_year_2900-3800_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_backward      = fh.variables['time_month'][:]                     #Model years
    #TEMP_backward       = fh.variables['TEMP_month'][:,220::,0:100]         #Sea level pressure (av\eraged over months) [hPa]
    TEMP_backward_1100       = fh.variables['TEMP_month'][:,220::,0:100]  
    
elif ts == 'yearly':
    time_backward      = fh.variables['time'][:]                     #Model years
    TEMP_backward_1100       = fh.variables['TEMP'][:,220::,0:100]           #Sea level pressure (av\eraged over months) [hPa]

fh.close()


fh      = netcdf.Dataset(directory_data+'VEL_Atlantic_depth_averaged_100_300m_year_600-1500_month_1-12_QE.nc', 'r')

time_forward       = fh.variables['time'][:] #Model years
lon_u                = fh.variables['lon'][:, :]  #Array of longitudes [degE]
lat_u                = fh.variables['lat'][:, :]  #Array of latitudes [degN]
UVEL_forward       = fh.variables['UVEL'][:, :, :]   #Annual zonal velocity
VVEL_forward       = fh.variables['VVEL'][:, :, :]   #Annual zonal velocity
WVEL_forward       = fh.variables['WVEL'][:, :, :]   #Annual zonal velocity
area_u               = fh.variables['area'][:]

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_Atlantic_depth_averaged_100_300m_year_2900-3800_month_1-12_QE.nc', 'r')

time_backward       = fh.variables['time'][:] #Model years
lon_u                 = fh.variables['lon'][:, :]  #Array of longitudes [degE]
lat_u                 = fh.variables['lat'][:, :]  #Array of latitudes [degN]
UVEL_backward       = fh.variables['UVEL'][:, :, :]   #Annual zonal velocity
VVEL_backward       = fh.variables['VVEL'][:, :, :]   #Annual zonal velocity
WVEL_backward       = fh.variables['WVEL'][:, :, :]   #Annual zonal velocity

fh.close()

#%% Meridionally average over certain latitude45 band

lat_min = 45
lat_max = 60

#Note that for higher latitudes the difference in lat per longitude becomes larger.
lat_min_index	= (fabs(lat[:,0] - lat_min)).argmin()
lat_max_index	= (fabs(lat[:,0] - lat_max)).argmin() + 1	

#Take variables in latitude section
lat_section             = lat[lat_min_index:lat_max_index,:]
area_section            = area[lat_min_index:lat_max_index,:]

lon_section             = lon[lat_min_index:lat_max_index,:]
SST_forward_section     = SST_forward[:,lat_min_index:lat_max_index,:]
SST_backward_section    = SST_backward[:,lat_min_index:lat_max_index,:]

TEMP_forward_section     = TEMP_forward_1100[:,lat_min_index:lat_max_index,:]
TEMP_backward_section    = TEMP_backward_1100[:,lat_min_index:lat_max_index,:]

lat_min_index_u	= (fabs(lat_u[:,0] - lat_min)).argmin()
lat_max_index_u	= (fabs(lat_u[:,0] - lat_max)).argmin() + 1	
lat_section_u             = lat_u[lat_min_index_u:lat_max_index_u,:]
area_section_u            = area_u[lat_min_index_u:lat_max_index_u,:]
lon_section_u             = lon_u[lat_min_index_u:lat_max_index_u,:]
UVEL_forward_section     = UVEL_forward[:,lat_min_index_u:lat_max_index_u,:]
UVEL_backward_section    = UVEL_backward[:,lat_min_index_u:lat_max_index_u,:]

plt.figure()
plt.contourf(lon_section, lat_section, SST_forward_section[0])

plt.figure()
plt.contourf(lon_section, lat_section, SST_backward_section[0])

plt.figure()
plt.contourf(lon_section, lat_section, area_section)

#plt.figure()
#plt.contourf(lon_section_u, lat_section_u, TEMP_forward_section[0])

lon_grid              = 2
lon_hovmoller         = np.arange(-60, 20.01, lon_grid)
SST_hovmoller_forward = ma.masked_all((len(SST_forward_section), len(lon_hovmoller)))
SST_hovmoller_backward = ma.masked_all((len(SST_backward_section), len(lon_hovmoller)))
TEMP_hovmoller_forward = ma.masked_all((len(TEMP_forward_section), len(lon_hovmoller)))
TEMP_hovmoller_backward = ma.masked_all((len(TEMP_backward_section), len(lon_hovmoller)))
UVEL_hovmoller_forward = ma.masked_all((len(UVEL_forward_section), len(lon_hovmoller)))
UVEL_hovmoller_backward = ma.masked_all((len(UVEL_backward_section), len(lon_hovmoller)))

for lon_i in range(len(lon_hovmoller)):
    #Remap all the data to standardised grid
    grid_index      = np.where((lon_hovmoller[lon_i] - lon_grid / 2.0 <= lon_section) & (lon_hovmoller[lon_i] + lon_grid / 2.0 > lon_section))
    area_hovmoller  = area_section[grid_index]
    
    if np.all(area_hovmoller.mask):
        #Only land surfaces, skip
        continue
    
    #Normalise, also adjust the area grid to match with the subsurface field
    SST_hovmoller_1         = SST_forward_section[:,grid_index[0],grid_index[1]]
    SST_hovmoller_2         = SST_backward_section[:,grid_index[0],grid_index[1]]
    TEMP_hovmoller_1        = TEMP_forward_section[:,grid_index[0],grid_index[1]]
    TEMP_hovmoller_2        = TEMP_backward_section[:,grid_index[0],grid_index[1]]
    area_hovmoller          = ma.masked_array(area_hovmoller, SST_hovmoller_1[0].mask)
    area_hovmoller          = area_hovmoller / np.sum(area_hovmoller)

    #Take the spatial mean, which is actual the meridional average
    SST_hovmoller_forward[:, lon_i] = np.sum(SST_hovmoller_1 * area_hovmoller, axis = 1)
    SST_hovmoller_backward[:, lon_i] = np.sum(SST_hovmoller_2 * area_hovmoller, axis = 1)
    TEMP_hovmoller_forward[:, lon_i] = np.sum(TEMP_hovmoller_1 * area_hovmoller, axis = 1)
    TEMP_hovmoller_backward[:, lon_i] = np.sum(TEMP_hovmoller_2 * area_hovmoller, axis = 1)

#%%

plt.figure(figsize=(8,6))
CS = plt.contourf(SST_hovmoller_1)


#%%

# Define the window size (100 years)
window_size = 100

if ts == 'monthly':
    window_size = window_size*12

# Create masked arrays for detrended data
SST_forward_detrend = ma.masked_all(SST_hovmoller_forward.shape)
SST_backward_detrend = ma.masked_all(SST_hovmoller_backward.shape)

TEMP_forward_detrend = ma.masked_all(TEMP_hovmoller_forward.shape)
TEMP_backward_detrend = ma.masked_all(TEMP_hovmoller_backward.shape)

# Detrend data over 100-year windows
for lon_i in range(len(lon_hovmoller)):
    for start in range(0, len(time_forward), window_size):
            # Define the end of the current window
            end = min(start + window_size, len(time_forward))  # Ensure we don't go out of bounds

            # Detrend the current window for SST_forward
            SST_forward_detrend[start:end, lon_i] = TrendRemover(
                time_forward[start:end], SST_hovmoller_forward[start:end, lon_i], 1)
            
            TEMP_forward_detrend[start:end, lon_i] = TrendRemover(
                time_forward[start:end], TEMP_hovmoller_forward[start:end, lon_i], 1)
            
    for start in range(0, len(time_backward), window_size):
            # Define the end of the current window
            end = min(start + window_size, len(time_backward))  # Ensure we don't go out of bounds

            # Detrend the current window for SST_forward
            SST_backward_detrend[start:end, lon_i] = TrendRemover(
                time_backward[start:end], SST_hovmoller_backward[start:end, lon_i], 1)
            
            TEMP_backward_detrend[start:end, lon_i] = TrendRemover(
                time_backward[start:end], TEMP_hovmoller_backward[start:end, lon_i], 1)
            
#%%

# --- choose bands ---
fs = 1.0 #if yearly, 12.0 if monthly
lowcut  = 1/70   # 70-year
highcut = 1/20   # 20-year

# --- preprocess ---
TEMP_f_DT = ma.masked_all((len(time_forward), len(lon_hovmoller)))
TEMP_b_DT = ma.masked_all((len(time_backward), len(lon_hovmoller)))

for lon_i in range(len(lon_hovmoller)):
    TEMP_f_DT[:, lon_i] = TrendRemover(time_forward, TEMP_hovmoller_forward[:, lon_i], 2)   
    TEMP_b_DT[:, lon_i] = TrendRemover(time_backward, TEMP_hovmoller_backward[:, lon_i], 2)

plt.figure(figsize=(8,6))
CS = plt.contourf(TEMP_hovmoller_forward)

plt.figure(figsize=(8,6))
CS = plt.contourf(TEMP_f_DT)

# optional: remove lon-mean at each time to emphasize propagation
TEMP_f_hp = TEMP_f_DT - ma.mean(TEMP_f_DT, axis=1)[:, None]
TEMP_b_hp = TEMP_b_DT - ma.mean(TEMP_b_DT, axis=1)[:, None]

# --- bandpass ---
TEMP_f_bp = bandpass_filter(TEMP_f_hp, lowcut, highcut, fs, order=4)
TEMP_b_bp = bandpass_filter(TEMP_b_hp, lowcut, highcut, fs, order=4)

# edge trim (recommended)
trim = 50
TEMP_f_bp = TEMP_f_bp[trim:-trim, :]
TEMP_b_bp = TEMP_b_bp[trim:-trim, :]


#%%Band pass filter

Tsub_forward_filtered = bandpass_filter(TEMP_forward_detrend, lowcut, highcut, fs)
Tsub_backward_filtered = bandpass_filter(TEMP_backward_detrend, lowcut, highcut, fs)

plt.figure(figsize=(8,6))
CS = plt.contourf(lon_hovmoller, time_forward, Tsub_forward_filtered, levels = np.linspace(-0.08, 0.08, 19), extend='both', cmap='RdBu_r')
plt.xlim(-60, 0)
plt.ylim(650,900)
cbar = plt.colorbar(CS)
cbar.set_label('Temperature [$^\circ$C]', fontsize=14)
plt.title('a) PI$^\mathrm{on}_{\mathrm{QE}}$ ('+str(lat_min)+'-'+str(lat_max)+'$^\circ$N)', fontsize=18)
plt.xlabel('Longitude $^\circ$E', fontsize=14)
plt.ylabel('Time [model year]', fontsize=14)
plt.tick_params(axis='both', labelsize=13)
plt.tight_layout()
plt.savefig(directory_figures +'Hovmoller-AMOC_on.pdf')

plt.figure(figsize=(8,6))
CS = plt.contourf(lon_hovmoller, time_backward, Tsub_backward_filtered, levels = np.linspace(-0.08, 0.08, 19), extend='both', cmap='RdBu_r')
plt.xlim(-60, 0)
plt.ylim(2950,3200)
cbar = plt.colorbar(CS)
cbar.set_label('Temperature [$^\circ$C]', fontsize=14)
plt.title('b) PI$^\mathrm{off}_{\mathrm{QE}}$ ('+str(lat_min)+'-'+str(lat_max)+'$^\circ$N)', fontsize=18)
plt.xlabel('Longitude $^\circ$E', fontsize=14)
plt.ylabel('Time [model year]', fontsize=14)
plt.tick_params(axis='both', labelsize=13)
plt.tight_layout()
plt.savefig(directory_figures +'Hovmoller-AMOC_off.pdf')

#%%

years_forward = np.arange(600, 1501)        
years_valid_forward = years_forward[trim:-trim]

years_backward = np.arange(2900, 3801)       
years_valid_backward = years_backward[trim:-trim]

plt.figure(figsize=(8,6))
CS = plt.contourf(lon_hovmoller, years_valid_forward, TEMP_f_bp, levels = np.linspace(-0.08, 0.08, 19), extend='both', cmap='RdBu_r')
plt.xlim(-60, 0)
plt.ylim(650,900)
cbar = plt.colorbar(CS)
cbar.set_label('Temperature [$^\circ$C]', fontsize=14)
plt.title('a) PI$^\mathrm{on}_{\mathrm{QE}}$ ('+str(lat_min)+'-'+str(lat_max)+'$^\circ$N)', fontsize=18)
plt.xlabel('Longitude $^\circ$E', fontsize=14)
plt.ylabel('Time [model year]', fontsize=14)
plt.tight_layout()

plt.figure(figsize=(8,6))
CS = plt.contourf(lon_hovmoller, years_valid_backward, TEMP_b_bp, levels = np.linspace(-0.08, 0.08, 19), extend='both', cmap='RdBu_r')
plt.xlim(-60, 0)
plt.ylim(2950,3200)
cbar = plt.colorbar(CS)
cbar.set_label('Temperature [$^\circ$C]', fontsize=14)
plt.title('b) AMOC off ('+str(lat_min)+'-'+str(lat_max)+'$^\circ$N)', fontsize=18)
plt.xlabel('Longitude $^\circ$E', fontsize=14)
plt.ylabel('Time [model year]', fontsize=14)
plt.tight_layout()

#%% Using a 10 year running mean

N = 10

if ts == 'monthly':
    N = N*12

SST_forward_mean = ma.masked_all((len(time_forward) - N + 1, len(lon_hovmoller)))
SST_backward_mean = ma.masked_all((len(time_backward) - N + 1, len(lon_hovmoller)))

TEMP_forward_mean = ma.masked_all((len(time_forward) - N + 1, len(lon_hovmoller)))
TEMP_backward_mean = ma.masked_all((len(time_backward) - N + 1, len(lon_hovmoller)))

for lon_i in range(len(lon_hovmoller)):
    SST_forward_mean[:,lon_i] = running_mean(SST_forward_detrend[:, lon_i], N)
    SST_backward_mean[:,lon_i] = running_mean(SST_backward_detrend[:, lon_i], N)
    TEMP_forward_mean[:,lon_i] = running_mean(TEMP_forward_detrend[:, lon_i], N)
    TEMP_backward_mean[:,lon_i] = running_mean(TEMP_backward_detrend[:, lon_i], N)
    
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

CS1 = axs[0].contourf(lon_hovmoller, time_forward[:-N+1], SST_forward_mean - np.mean(SST_forward_mean[0:100], axis = 0), levels=np.linspace(-0.5, 0.5, 19), extend='both', cmap='RdBu_r')
axs[0].set_xlim(-60, 0)
axs[0].set_ylim(600, 800)
axs[0].set_title('a) AMOC on (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[0].set_xlabel('Longitude $^\circ$E')
axs[0].set_ylabel('Time [model year]')
cbar1 = fig.colorbar(CS1, ax=axs[0], orientation='vertical')
cbar1.set_label('Temperature anomaly [$^\circ$C]')

CS2 = axs[1].contourf(lon_hovmoller, time_backward[:-N+1], SST_backward_mean - np.mean(SST_backward_mean[0:100], axis = 0), levels=np.linspace(-0.5, 0.5, 19), extend='both', cmap='RdBu_r')
axs[1].set_xlim(-60, 0)
axs[1].set_ylim(2900, 3100)
axs[1].set_title('b) AMOC off (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[1].set_xlabel('Longitude $^\circ$E')
cbar2 = fig.colorbar(CS2, ax=axs[1], orientation='vertical')
cbar2.set_label('Temperature anomaly [$^\circ$C]')

plt.tight_layout()
plt.savefig(directory_figures + 'Hovmoller_TEMP_depth_averaged_100-300m_lat_'+str(lat_min)+'-'+str(lat_max)+'_AMOC_on_off_QE_runningmean_'+str(N)+'.pdf')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

CS1 = axs[0].contourf(lon_hovmoller, time_forward[:-N+1], SST_forward_mean - np.mean(SST_forward_mean[0:100], axis = 0), levels=np.linspace(-0.5, 0.5, 19), extend='both', cmap='RdBu_r')
axs[0].set_xlim(-60, 0)
axs[0].set_ylim(1300, 1500)
axs[0].set_title('c) AMOC on (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[0].set_xlabel('Longitude $^\circ$E')
axs[0].set_ylabel('Time [model year]')
cbar1 = fig.colorbar(CS1, ax=axs[0], orientation='vertical')
cbar1.set_label('Temperature anomaly [$^\circ$C]')

CS2 = axs[1].contourf(lon_hovmoller, time_backward[:-N+1], SST_backward_mean - np.mean(SST_backward_mean[0:100], axis = 0), levels=np.linspace(-0.5, 0.5, 19), extend='both', cmap='RdBu_r')
axs[1].set_xlim(-60, 0)
axs[1].set_ylim(3600, 3800)
axs[1].set_title('d) AMOC off (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[1].set_xlabel('Longitude $^\circ$E')
cbar2 = fig.colorbar(CS2, ax=axs[1], orientation='vertical')
cbar2.set_label('Temperature anomaly [$^\circ$C]')

plt.tight_layout()
plt.savefig(directory_figures + 'Hovmoller_TEMP_depth_averaged_100-300m_lat_'+str(lat_min)+'-'+str(lat_max)+'_AMOC_on_off_QE_runningmean_'+str(N)+'_2.pdf')
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

CS1 = axs[0].contourf(lon_hovmoller, time_forward[:-N+1], TEMP_forward_mean - np.mean(TEMP_forward_mean[0:100], axis = 0), levels=np.linspace(-0.3, 0.3, 19), extend='both', cmap='RdBu_r')
axs[0].set_xlim(-60, 0)
axs[0].set_ylim(600, 800)
axs[0].set_title('a) AMOC on (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[0].set_xlabel('Longitude $^\circ$E')
axs[0].set_ylabel('Time [model year]')
cbar1 = fig.colorbar(CS1, ax=axs[0], orientation='vertical')
cbar1.set_label('Temperature anomaly [$^\circ$C]')

CS2 = axs[1].contourf(lon_hovmoller, time_backward[:-N+1], TEMP_backward_mean - np.mean(TEMP_backward_mean[0:100], axis = 0), levels=np.linspace(-0.3, 0.3, 19), extend='both', cmap='RdBu_r')
axs[1].set_xlim(-60, 0)
axs[1].set_ylim(2900, 3100)
axs[1].set_title('b) AMOC off (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[1].set_xlabel('Longitude $^\circ$E')
cbar2 = fig.colorbar(CS2, ax=axs[1], orientation='vertical')
cbar2.set_label('Temperature anomaly [$^\circ$C]')

plt.tight_layout()
plt.savefig(directory_figures + 'Hovmoller_TEMP_depth_averaged_0-1100m_lat_'+str(lat_min)+'-'+str(lat_max)+'_AMOC_on_off_QE_runningmean_'+str(N)+'.pdf')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

CS1 = axs[0].contourf(lon_hovmoller, time_forward[:-N+1], TEMP_forward_mean - np.mean(TEMP_forward_mean[0:100], axis = 0), levels=np.linspace(-0.3, 0.3, 19), extend='both', cmap='RdBu_r')
axs[0].set_xlim(-60, 0)
axs[0].set_ylim(1300, 1500)
axs[0].set_title('c) AMOC on (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[0].set_xlabel('Longitude $^\circ$E')
axs[0].set_ylabel('Time [model year]')
cbar1 = fig.colorbar(CS1, ax=axs[0], orientation='vertical')
cbar1.set_label('Temperature anomaly [$^\circ$C]')

CS2 = axs[1].contourf(lon_hovmoller, time_backward[:-N+1], TEMP_backward_mean - np.mean(TEMP_backward_mean[0:100], axis = 0), levels=np.linspace(-0.3, 0.3, 19), extend='both', cmap='RdBu_r')
axs[1].set_xlim(-60, 0)
axs[1].set_ylim(3600, 3800)
axs[1].set_title('d) AMOC off (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[1].set_xlabel('Longitude $^\circ$E')
cbar2 = fig.colorbar(CS2, ax=axs[1], orientation='vertical')
cbar2.set_label('Temperature anomaly [$^\circ$C]')

plt.tight_layout()
plt.savefig(directory_figures + 'Hovmoller_TEMP_depth_averaged_0-1100m_lat_'+str(lat_min)+'-'+str(lat_max)+'_AMOC_on_off_QE_runningmean_'+str(N)+'_2.pdf')
plt.show()

#%%

UVEL_forward_mean = ma.masked_all((len(time_forward) - N + 1, len(lon_u[0])))
UVEL_backward_mean = ma.masked_all((len(time_backward) - N + 1, len(lon_u[0])))

for lon_i in range(len(lon_u[0])):
    UVEL_forward_mean[:,lon_i] = running_mean(UVEL_forward_av[:, lon_i], N)
    UVEL_backward_mean[:,lon_i] = running_mean(UVEL_backward_av[:, lon_i], N)
    
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

CS1 = axs[0].contourf(lon_u[0], time_forward[:-N+1], UVEL_forward_mean - np.mean(UVEL_forward_mean[0:100]), levels=np.linspace(-0.002, 0.002, 19), extend='both', cmap='RdBu_r')
axs[0].set_xlim(-60, 0)
axs[0].set_ylim(600, 900)
axs[0].set_title('a) AMOC on (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[0].set_xlabel('Longitude $^\circ$E')
axs[0].set_ylabel('Time [model year]')
cbar1 = fig.colorbar(CS1, ax=axs[0], orientation='vertical')
cbar1.set_label('Temperature anomaly [$^\circ$C]')

CS2 = axs[1].contourf(lon_u[0], time_backward[:-N+1], UVEL_backward_mean - np.mean(UVEL_backward_mean[0:100]), levels=np.linspace(-0.002, 0.002, 19), extend='both', cmap='RdBu_r')
axs[1].set_xlim(-60, 0)
axs[1].set_ylim(2900, 3200)
axs[1].set_title('b) AMOC off (' + str(lat_min) + '-' + str(lat_max) + '$^\circ$N)')
axs[1].set_xlabel('Longitude $^\circ$E')
cbar2 = fig.colorbar(CS2, ax=axs[1], orientation='vertical')
cbar2.set_label('Temperature anomaly [$^\circ$C]')

plt.tight_layout()
plt.savefig(directory_figures + 'Hovmoller_UVEL_depth_averaged_100-300m_lat_'+str(lat_min)+'-'+str(lat_max)+'_AMOC_on_off_QE_runningmean_'+str(N)+'.pdf')
plt.show()



