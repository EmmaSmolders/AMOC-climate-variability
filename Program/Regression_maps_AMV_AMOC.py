#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 12:36:28 2026

@author: 6008399

Regression patterns of SST or TEMP on the AMV or AMOC index

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
from scipy.signal import detrend
from scipy.ndimage import uniform_filter1d
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import dpss

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%% Functions

def running_mean_nan_1d(x, window, center=True):
    """
    Running mean for 1D series with NaNs.
    Returns an array of the SAME length as x (NaN at edges if center=True).
    """
    x = np.asarray(x, float)
    n = x.size
    if window is None or window <= 1:
        return x.copy()

    out = np.full(n, np.nan)

    if center:
        half = window // 2
        for i in range(n):
            i0 = max(0, i - half)
            i1 = min(n, i + half + (0 if window % 2 == 0 else 1))
            # For even windows, this gives a slightly left-centered window; that's fine for an index
            seg = x[i0:i1]
            if np.isfinite(seg).any():
                out[i] = np.nanmean(seg)
    else:
        for i in range(window - 1, n):
            seg = x[i - window + 1:i + 1]
            if np.isfinite(seg).any():
                out[i] = np.nanmean(seg)

    return out

def detrend_poly_1d(time, x, order=2):
    """
    Polynomial detrend for 1D array with NaNs.
    Returns detrended series with NaNs preserved.
    """
    t = np.asarray(time, float)
    y = np.asarray(x, float)

    good = np.isfinite(t) & np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=float)
    if good.sum() < max(10, order + 2):
        return out

    p = np.polyfit(t[good], y[good], order)
    fit = np.polyval(p, t)
    out[good] = y[good] - fit[good]
    return out

def TrendRemover(time, data, trend_type):
	"""Removes trend of choice, 1 = linear, 2 = quadratic, etc."""
	
	rank 	= polyfit(time, data, trend_type)
	fitting = np.zeros(len(time))
		
	for rank_i in range(len(rank)):
		#Get the least-squre fit
		fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))

	#Subtract the fitted output
	data -= fitting
	
	return data

def north_atlantic_mask(lat, lon, lat_min, lat_max):
    """
    Boolean mask for the North Atlantic (2D curvilinear grid).

    Parameters
    ----------
    lat, lon : np.ndarray (2D)
        Latitude and longitude in degrees.

    Returns
    -------
    mask : np.ndarray (bool, 2D)
        True where grid cell is inside the North Atlantic region.
    """

    lat_min, lat_max = lat_min, lat_max
    lon_min, lon_max = -75.0, 0

    # Handle 0–360 longitude grids
    if np.nanmin(lon) >= 0:
        lon_min = (lon_min + 360) % 360
        lon_max = (lon_max + 360) % 360

    mask = (lat >= lat_min) & (lat <= lat_max)

    if lon_min < lon_max:
        mask &= (lon >= lon_min) & (lon <= lon_max)
    else:
        # Region crosses prime meridian (0°)
        mask &= (lon >= lon_min) | (lon <= lon_max)

    return mask

def compute_amv_index_eof_consistent(
    FIELD,          # (time, lat, lon): monthly SST or TEMP
    time,           # (time,): e.g., 2900.0, 2900.0833, ...
    lat2d, lon2d, area2d, # (lat, lon) 2D grids
    lat_min=0, lat_max=60,
    basin_mask_func=None, # function(lat2d, lon2d, lat_min, lat_max)->bool mask (True in region)
    lowpass=10, #in years
    detrend_order=2,
    remove_monthly_clim=False,
    standardize=True
):
    """
    EOF-consistent AMV basin-mean index:
      1) (optional) remove monthly climatology at each grid point
      2) area-weighted NA mean time series
      3) detrend (order=2 by default) on the MONTHLY series using time_month
      4) 120-month running-mean low-pass (centered)
      5) (optional) standardize (z-score)

    Returns:
      amv      : processed index (same length as input time; NaNs near edges from lowpass)
      na_mean  : raw area-weighted NA mean after (optional) climatology removal
      na_dt    : detrended NA mean (before lowpass)
      bg_lp    : lowpassed detrended series (before standardize)
    """
    # --- input as masked arrays where possible ---
    F = ma.array(FIELD, copy=False)
    lat2d = ma.array(lat2d, copy=False)
    lon2d = ma.array(lon2d, copy=False)
    area2d = ma.array(area2d, copy=False)

    ntime, nlat, nlon = F.shape

    # --- region mask ---
    if basin_mask_func is None:
        # very simple default: just latitude band (no land/med masks)
        region = (lat2d >= lat_min) & (lat2d <= lat_max)
    else:
        region = basin_mask_func(lat2d, lon2d, lat_min, lat_max)

    # ensure region is boolean ndarray
    region = np.asarray(region, dtype=bool)

    # --- apply land mask from first timestep if FIELD is masked ---
    landmask = ma.getmaskarray(F[0])
    w = ma.array(area2d, mask=(~region) | landmask)

    # --- optional: remove monthly climatology at each grid point ---
    if remove_monthly_clim:
        # subtract 12-month climatology per grid cell (masked-safe)
        F2 = F.copy()
        for m in range(12):
            idx = np.arange(m, ntime, 12)
            clim = F2[idx].mean(axis=0)
            F2[idx] = F2[idx] - clim
        F_use = F2
    else:
        F_use = F

    # --- area-weighted basin mean time series ---
    # masked-safe sums
    num = (F_use * w[None, :, :]).sum(axis=(1, 2))
    den = w.sum()
    na_mean = (num / den).filled(np.nan)

    # --- detrend order-2 on monthly series ---
    na_dt = detrend_poly_1d(time, na_mean, order=detrend_order)

    # --- 120-month low-pass on detrended series ---
    bg_lp = running_mean_nan_1d(na_dt, window=lowpass, center=True)

    # --- standardize if requested ---
    if standardize:
        mu = np.nanmean(bg_lp)
        sig = np.nanstd(bg_lp)
        amv = (bg_lp - mu) / sig if np.isfinite(sig) and sig > 0 else bg_lp * np.nan
    else:
        amv = bg_lp

    return amv, na_mean, na_dt, bg_lp

def make_regression_files(field, index):
    mask_t = np.isfinite(index)
    field = field[mask_t, :, :]
    idx   = index[mask_t]

    # standardize index (nan-safe)
    idx = (idx - np.nanmean(idx)) / np.nanstd(idx)

    # anomalies
    field_anom = field - np.nanmean(field, axis=0)

    # regression = cov(field, idx_std)
    return np.nanmean(field_anom * idx[:, None, None], axis=0)


#%% Read in data

#Choose which months you want (i.e. DJF or JJA)
month_start = 1
month_end   = 12

ts = 'yearly' #or 'montly' or 'yearly'

#fh      = netcdf.Dataset(directory_data+'TEMP_month_'+str(month_start)+'-'+str(month_end)+'_QE_year_0-2200.nc', 'r')

fh       = netcdf.Dataset(directory_data + 'TEMP_Atlantic_depth_averaged_100_300m_year_600-1500_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_forward      = fh.variables['time_month'][:]                     #Model years
    TEMP_forward       = fh.variables['TEMP_month'][:,220::,:]          #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_forward      = fh.variables['time'][:]                     #Model years
    TEMP_forward       = fh.variables['TEMP'][:,220::,:]        #Sea level pressure (av\eraged over months) [hPa]

    
lon               = fh.variables['lon'][220::,:]             #Array of longitudes [degE]
lat               = fh.variables['lat'][220::,:]             #Array of latitudes [degN]
area              = fh.variables['area'][220::,:] 

fh.close()

fh       = netcdf.Dataset(directory_data + 'TEMP_Atlantic_depth_averaged_100_300m_year_2900-3800_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_backward      = fh.variables['time_month'][:]                     #Model years
    #TEMP_backward       = fh.variables['TEMP_month'][:,220::,0:100]         #Sea level pressure (av\eraged over months) [hPa]
    TEMP_backward       = fh.variables['TEMP_month'][:,220::,:] 
    
elif ts == 'yearly':
    time_backward      = fh.variables['time'][:]                     #Model years
    TEMP_backward       = fh.variables['TEMP'][:,220::,:]          #Sea level pressure (av\eraged over months) [hPa]
fh.close()

fh       = netcdf.Dataset(directory_data + 'SST_Atlantic_year_600-1500_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_forward      = fh.variables['time_month'][:]                     #Model years
    SST_forward       = fh.variables['SST_month'][:,220::,:]          #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_forward      = fh.variables['time'][:]                     #Model years
    SST_forward       = fh.variables['SST'][:,220::,:]        #Sea level pressure (av\eraged over months) [hPa]
fh.close()

fh       = netcdf.Dataset(directory_data + 'SST_Atlantic_year_2900-3800_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_backward      = fh.variables['time_month'][:]                     #Model years
    SST_backward       = fh.variables['SST_month'][:,220::,:]          #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_backward      = fh.variables['time'][:]                     #Model years
    SST_backward       = fh.variables['SST'][:,220::,:]          #Sea level pressure (av\eraged over months) [hPa]
fh.close()

plt.figure()
plt.contourf(lon, lat, TEMP_forward[0])

plt.figure()
plt.contourf(lon, lat, SST_forward[0])

#%%

plt.figure()
plt.plot(np.mean(SST_forward, axis=(1,2)))

plt.figure()
plt.plot(np.mean(TEMP_forward, axis=(1,2)))

plt.figure()
plt.plot(np.mean(SST_backward, axis=(1,2)))

plt.figure()
plt.contourf(lon, lat, np.mean(SST_backward, axis=0) - np.mean(SST_forward, axis=0), cmap='RdBu_r', levels=np.linspace(-15, 15, 21))
plt.colorbar()

plt.figure()
plt.contourf(lon, lat, np.mean(TEMP_backward, axis=0) - np.mean(TEMP_forward, axis=0), cmap='RdBu_r', levels=np.linspace(-15, 15, 21))
plt.colorbar()

#%%

SST_forward = ma.array(SST_forward).filled(np.nan)
TEMP_forward = ma.array(TEMP_forward).filled(np.nan)
SST_backward = ma.array(SST_backward).filled(np.nan)
TEMP_backward = ma.array(TEMP_backward).filled(np.nan)


#%%

lat_min = 0
lat_max = 60

amv_sst_forward, sst_mean_forward, sst_dt_forward, sst_lp_forward = compute_amv_index_eof_consistent(
    SST_forward, time_forward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=10,
    detrend_order=2,
    remove_monthly_clim=False,
    standardize=False)

amv_sst_backward, sst_mean_backward, sst_dt_backward, sst_lp_backward = compute_amv_index_eof_consistent(
    SST_backward, time_backward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=10,
    detrend_order=2,
    remove_monthly_clim=False,
    standardize=False)

amv_temp_forward, temp_mean_forward, temp_dt_forward, temp_lp_forward = compute_amv_index_eof_consistent(
    TEMP_forward, time_forward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=10,
    detrend_order=2,
    remove_monthly_clim=False,
    standardize=False)

amv_temp_backward, temp_mean_backward, temp_dt_backward, temp_lp_backward = compute_amv_index_eof_consistent(
    TEMP_backward, time_backward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=10,
    detrend_order=2,
    remove_monthly_clim=False,
    standardize=False)

#%% Pointwise detrending of SST and TEMP

SST_DT_forward = ma.masked_all(SST_forward.shape)
TEMP_DT_forward = ma.masked_all(TEMP_forward.shape)
SST_DT_backward = ma.masked_all(SST_backward.shape)
TEMP_DT_backward = ma.masked_all(TEMP_backward.shape)

SST_DT_lp_forward = ma.masked_all(SST_forward.shape)
TEMP_DT_lp_forward = ma.masked_all(TEMP_forward.shape)
SST_DT_lp_backward = ma.masked_all(SST_backward.shape)
TEMP_DT_lp_backward = ma.masked_all(TEMP_backward.shape)

for lat_i in range(len(lat)):
    for lon_i in range(len(lon[0])):
        SST_DT_forward[:, lat_i, lon_i] = TrendRemover(time_forward, SST_forward[:, lat_i, lon_i].copy(), 2)
        SST_DT_backward[:, lat_i, lon_i] = TrendRemover(time_backward, SST_backward[:, lat_i, lon_i].copy(), 2)
        TEMP_DT_forward[:, lat_i, lon_i] = TrendRemover(time_forward, TEMP_forward[:, lat_i, lon_i].copy(), 2)
        TEMP_DT_backward[:, lat_i, lon_i] = TrendRemover(time_backward, TEMP_backward[:, lat_i, lon_i].copy(), 2)

        SST_DT_lp_forward[:, lat_i, lon_i] = running_mean_nan_1d(SST_DT_forward[:, lat_i, lon_i], window = 10, center=True)
        SST_DT_lp_backward[:, lat_i, lon_i] = running_mean_nan_1d(SST_DT_backward[:, lat_i, lon_i], window = 10, center=True)
        TEMP_DT_lp_forward[:, lat_i, lon_i] = running_mean_nan_1d(TEMP_DT_forward[:, lat_i, lon_i], window = 10, center=True)
        TEMP_DT_lp_backward[:, lat_i, lon_i] = running_mean_nan_1d(TEMP_DT_backward[:, lat_i, lon_i], window = 10, center=True)

#%%

lat_idx_rapid   = 190
lat_idx_45N     = 226
lat_idx         = lat_idx_rapid

fh_transient = netcdf.Dataset(r'/Users/6008399/Documents/PhD/CESM_collapse/netcdf/AMOC_max_year_0-2200.nc','r')

time_forward        = fh_transient.variables['time'][599:1500]     #time in model years
AMOC_forward        = fh_transient.variables['AMOC_max'][599:1500, lat_idx] #Volume transport [Sv]
lat_amoc            = fh_transient.variables['lat'][lat_idx]

print(lat_amoc)

fh_transient.close()

fh_transient = netcdf.Dataset(r'/Users/6008399/Documents/PhD/CESM_collapse/netcdf/AMOC_max_QE_year_2201-4400.nc','r')

time_backward        = fh_transient.variables['time'][699:1600]     #time in model years
AMOC_backward        = fh_transient.variables['AMOC_max'][699:1600, lat_idx] #Volume transport [Sv]

fh_transient.close()

#Detrend AMOC quadratically
AMOC_dt_forward = TrendRemover(time_forward, AMOC_forward.copy(), 2)
AMOC_dt_backward = TrendRemover(time_backward, AMOC_backward.copy(), 2)

plt.figure()
plt.plot(AMOC_forward)
plt.plot(AMOC_dt_forward)

#%%

plt.figure()
plt.contourf(SST_DT_forward[0])

plt.figure()
plt.contourf(TEMP_DT_forward[0])

#%%

regr_SST_AMVsst_forward = make_regression_files(SST_DT_lp_forward, amv_sst_forward)
regr_SST_AMVsst_backward = make_regression_files(SST_DT_lp_backward, amv_sst_backward)

regr_TEMP_AMVtemp_forward = make_regression_files(TEMP_DT_lp_forward, amv_temp_forward)
regr_TEMP_AMVtemp_backward = make_regression_files(TEMP_DT_lp_backward, amv_temp_backward)

regr_SST_AMVtemp_forward = make_regression_files(SST_DT_lp_forward, amv_temp_forward)
regr_SST_AMVtemp_backward = make_regression_files(SST_DT_lp_backward, amv_temp_backward)

regr_TEMP_AMVsst_forward = make_regression_files(TEMP_DT_lp_forward, amv_sst_forward)
regr_TEMP_AMVsst_backward = make_regression_files(TEMP_DT_lp_backward, amv_sst_backward)

regr_SST_AMOC_forward = make_regression_files(SST_DT_forward, AMOC_dt_forward)
regr_SST_AMOC_backward = make_regression_files(SST_DT_backward, AMOC_dt_backward)

regr_TEMP_AMOC_forward = make_regression_files(TEMP_DT_lp_forward, AMOC_dt_forward)
regr_TEMP_AMOC_backward = make_regression_files(TEMP_DT_lp_backward, AMOC_dt_backward)


#%%

regr_AMO_forward = make_regression_files(field=SST_DT_forward,
                                       index=amv_sst_forward)

regr_AMO_backward = make_regression_files(field=SST_DT_backward,
                                       index=amv_sst_backward)

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0].contourf(lon, lat, regr_AMO_forward, transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0], orientation='horizontal')
axs[0].set_title('a) Regression pattern AMOC on')
axs[0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0].xaxis.set_major_formatter(lon_formatter)
axs[0].set_yticks(np.arange(20,70,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1].contourf(lon, lat, regr_AMO_backward, transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1], orientation='horizontal')
axs[1].set_title('b) Regression pattern AMOC off')
axs[1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1].xaxis.set_major_formatter(lon_formatter)
axs[1].set_yticks(np.arange(20,70,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1].yaxis.set_major_formatter(lat_formatter)

plt.suptitle('Monthly data, piecewise quadratic detrending')
plt.tight_layout()
#plt.savefig(directory_figures + 'Regressionpattern_monthly_AMV_piecewise_quadratic.pdf')

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0].contourf(lon, lat, regr_SST_AMVsst_forward, transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0], orientation='horizontal')
axs[0].set_title('a) Regression pattern AMOC on')
axs[0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0].xaxis.set_major_formatter(lon_formatter)
axs[0].set_yticks(np.arange(20,70,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1].contourf(lon, lat, regr_SST_AMVsst_backward, transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1], orientation='horizontal')
axs[1].set_title('b) Regression pattern AMOC off')
axs[1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1].xaxis.set_major_formatter(lon_formatter)
axs[1].set_yticks(np.arange(20,70,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1].yaxis.set_major_formatter(lat_formatter)

plt.suptitle('Monthly data, piecewise quadratic detrending')
plt.tight_layout()
#plt.savefig(directory_figures + 'Regressionpattern_monthly_AMV_piecewise_quadratic.pdf')

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0].contourf(lon, lat, regr_TEMP_AMVsst_forward, transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0], orientation='horizontal')
axs[0].set_title('a) Regression pattern AMOC on')
axs[0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0].xaxis.set_major_formatter(lon_formatter)
axs[0].set_yticks(np.arange(20,70,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1].contourf(lon, lat, regr_TEMP_AMVsst_backward, transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1], orientation='horizontal')
axs[1].set_title('b) Regression pattern AMOC off')
axs[1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1].xaxis.set_major_formatter(lon_formatter)
axs[1].set_yticks(np.arange(20,70,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1].yaxis.set_major_formatter(lat_formatter)

plt.suptitle('Monthly data, piecewise quadratic detrending')
plt.tight_layout()
#plt.savefig(directory_figures + 'Regressionpattern_monthly_AMV_piecewise_quadratic.pdf')

#%%

fig, axs = plt.subplots(
    2, 2, figsize=(14, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# -----------------------
# Formatting helper
# -----------------------
for ax in axs.flat:
    ax.coastlines()
    ax.set_xticks(np.arange(-90, 31, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(20, 70, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

# =========================
# (a) SST – AMOC on
# =========================
c1 = axs[0,0].contourf(
    lon, lat, regr_SST_AMVsst_forward,
    levels=np.linspace(-0.4, 0.4, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree())
axs[0,0].set_title('a) SST regressed on AMV-sst (PI$^{on}_{QE}$)')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal',shrink=0.8,label='R [K Sv$^{-1}$]')
axs[0,0].set_ylim(10, 70)

# =========================
# (b) SST – AMOC off
# =========================
c2 = axs[0,1].contourf(
    lon, lat, regr_SST_AMVsst_backward,
    levels=np.linspace(-0.4, 0.4, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree())
axs[0,1].set_title('b) SST regressed on AMV-sst (PI$^{off}_{QE}$)')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal', shrink=0.8,label='R [K Sv$^{-1}$]')
axs[0,1].set_ylim(10, 70)

# =========================
# (c) TEMP – AMOC on
# =========================
c3 = axs[1,0].contourf(
    lon, lat, regr_TEMP_AMVsst_forward,
    levels=np.linspace(-0.4, 0.4, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree())
axs[1,0].set_title('c) TEMP (100–300m) regressed on AMV-sst (PI$^{on}_{QE}$)')
fig.colorbar(c3, ax=axs[1,0], orientation='horizontal',shrink=0.8,label='R [K Sv$^{-1}$]')
axs[1,0].set_ylim(10, 70)

# =========================
# (d) TEMP – AMOC off
# =========================
c4 = axs[1,1].contourf(
    lon, lat, regr_TEMP_AMVsst_backward,
    levels=np.linspace(-0.4, 0.4, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree())
axs[1,1].set_title('d) TEMP (100–300m) regressed on AMV-sst (PI$^{off}_{QE}$)')
fig.colorbar(c4, ax=axs[1,1], orientation='horizontal',shrink=0.8,label='R [K Sv$^{-1}$]')
axs[1,1].set_ylim(10, 70)

#plt.suptitle('Regression of SST and Subsurface Temperature onto AMV (sst) Strength',fontsize=14)

plt.tight_layout()
plt.savefig(directory_figures + 'Regression_SST_TEMP_AMV.pdf')
plt.show()

#%%

fig, axs = plt.subplots(
    2, 2, figsize=(14, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# -----------------------
# Formatting helper
# -----------------------
for ax in axs.flat:
    ax.coastlines()
    ax.set_xticks(np.arange(-90, 31, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(20, 70, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

# =========================
# (a) SST – AMOC on
# =========================
c1 = axs[0,0].contourf(
    lon, lat, regr_SST_AMOC_forward,
    levels=np.linspace(-0.3, 0.3, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree()
)
axs[0,0].set_title('a) SST regressed on AMOC 26$^\circ$N (PI$^{\mathrm{on}}_{\mathrm{QE}}$)')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal',shrink=0.8,
             label='R [K Sv$^{-1}$]')
axs[0,0].set_ylim(10, 70)
axs[0,0].set_xlim(-90, 10)

# =========================
# (b) SST – AMOC off
# =========================
c2 = axs[0,1].contourf(
    lon, lat, regr_SST_AMOC_backward,
    levels=np.linspace(-0.3, 0.3, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree()
)
axs[0,1].set_title('b) SST regressed on AMOC 26$^\circ$N (PI$^{\mathrm{off}}_{\mathrm{QE}}$)')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal', shrink=0.8,
             label='R [K Sv$^{-1}$]')
axs[0,1].set_ylim(10, 70)
axs[0,1].set_xlim(-90, 10)

# =========================
# (c) TEMP – AMOC on
# =========================
c3 = axs[1,0].contourf(
    lon, lat, regr_TEMP_AMOC_forward,
    levels=np.linspace(-0.1, 0.1, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree()
)
axs[1,0].set_title('c) TEMP (100–300m) regressed on AMOC 26$^\circ$N (PI$^{\mathrm{on}}_{\mathrm{QE}}$)')
fig.colorbar(c3, ax=axs[1,0], orientation='horizontal',shrink=0.8,
             label='R [K Sv$^{-1}$]')
axs[1,0].set_ylim(10, 70)
axs[1,0].set_xlim(-90, 10)

# =========================
# (d) TEMP – AMOC off
# =========================
c4 = axs[1,1].contourf(
    lon, lat, regr_TEMP_AMOC_backward,
    levels=np.linspace(-0.1, 0.1, 21),
    cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree()
)
axs[1,1].set_title('d) TEMP (100–300m) regressed on AMOC 26$^\circ$N (PI$^{\mathrm{off}}_{\mathrm{QE}}$)')
fig.colorbar(c4, ax=axs[1,1], orientation='horizontal',shrink=0.8,
             label='R [K Sv$^{-1}$]')
axs[1,1].set_ylim(10, 70)
axs[1,1].set_xlim(-90, 10)

#plt.suptitle('Regression of SST and Subsurface Temperature onto AMOC Strength',fontsize=14)

plt.tight_layout()
plt.savefig(directory_figures + 'Regression_SST_TEMP_AMOC_26N.pdf')
plt.show()


# %%
