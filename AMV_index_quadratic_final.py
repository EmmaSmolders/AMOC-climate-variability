#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:01:06 2026

@author: 6008399

EOF-consistent AMV index (quadratic detrending and low pass filter of NA basin mean temperatures)

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
#from spectrum import pmtm
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import dpss

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%%

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

def zscore_nan_1d(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 10:
        return x * np.nan
    mu = np.nanmean(x[m])
    sig = np.nanstd(x[m], ddof=1)
    return (x - mu) / sig

def ar1_coeff(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = (x - x.mean()) / x.std(ddof=1)
    a1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    return np.clip(a1, -0.99, 0.99)

def Welch(x, nperseg=256, nsurr=2000, ci=(95,), seed=0):
    """
    Annual data: fs=1/year. Returns f [1/yr], Pxx, ci_dict, a1
    CI is computed from AR(1) Monte-Carlo surrogates using the SAME Welch settings.
    """
    rng = np.random.default_rng(seed)

    x = zscore_nan_1d(x)
    x = x[np.isfinite(x)]

    fs = 1.0
    nperseg = min(nperseg, len(x))
    noverlap = nperseg // 2

    # Data PSD
    f, Pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)

    # AR(1) surrogates
    a1 = ar1_coeff(x)
    var = np.var(x)
    b = np.sqrt((1.0 - a1**2) * var)

    P_surr = np.zeros((nsurr, len(f)))
    spin = 200

    for s in range(nsurr):
        y = np.zeros(len(x))
        state = 0.0
        white = rng.normal(0, 1, spin + len(x))
        for i in range(spin + len(x)):
            state = a1 * state + b * white[i]
            if i >= spin:
                y[i - spin] = state

        y = (y - y.mean()) / y.std(ddof=1)
        _, Pyy = welch(y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
        P_surr[s, :] = Pyy

    ci_dict = {p: np.percentile(P_surr, p, axis=0) for p in ci}
    return f, Pxx, ci_dict, a1

def ReadinData(filename):

	fh = netcdf.Dataset(filename, 'r')

	lon 		= fh.variables['lon'][:]			#Longitude
	lat 		= fh.variables['lat'][:]			#Latitude 
	eof         = fh.variables['eof'][:]           #number of EOFs
	time		= fh.variables['time'][:]			#Model year
	PC		    = fh.variables['PC'][:] 			#Principal component
	VAR		    = fh.variables['VAR'][:]	 		#Variance of the PCs/EOFs
	EOF	       	= fh.variables['EOF'][:]			#EOFs

	fh.close()

	return lon, lat, eof, time, PC, VAR, EOF

def zscore_nan_1d(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 10:
        return x * np.nan
    mu = np.nanmean(x[m])
    sig = np.nanstd(x[m], ddof=1)
    return (x - mu) / sig

def ar1_phi(x):
    """Lag-1 autocorrelation estimate (clipped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = (x - x.mean()) / x.std(ddof=1)
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    return float(np.clip(phi, -0.99, 0.99))

def mtm_psd(x, fs=1.0, NW=2.0, Kmax=None):
    """
    Multitaper PSD using DPSS tapers.
    Returns f (0..Nyquist) and one-sided PSD estimate.
    x should be 1D, finite, ideally standardized.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 16:
        raise ValueError("Time series too short for MTM.")

    if Kmax is None:
        # Common choice: Kmax = int(2*NW) - 1
        Kmax = max(1, int(2*NW) - 1)

    # DPSS tapers
    tapers = dpss(n, NW, Kmax, return_ratios=False)  # (Kmax, n)

    # Apply tapers and FFT
    Xk = np.fft.rfft(tapers * x[None, :], axis=1)    # (Kmax, nfreq)
    Sk = (np.abs(Xk)**2)                             # raw eigenspectra

    # Simple average across tapers (equal weights)
    S = Sk.mean(axis=0)

    # Frequency vector
    f = np.fft.rfftfreq(n, d=1.0/fs)

    # Scale to "PSD-like" units so that integral approx ~ variance.
    # This scaling is consistent enough for comparing on/off and for MC envelopes.
    S = S / (fs * n)

    return f, S

def mtm_psd_ar1_ci(x, fs=1.0, NW=2.0, Kmax=None, nsurr=2000, ci=(90,95,99), seed=0):
    """
    MTM PSD of data + AR(1) surrogate percentile envelopes, computed using the same MTM settings.
    Returns f, S_data, ci_dict, phi
    """
    rng = np.random.default_rng(seed)

    xz = zscore_nan_1d(x)
    xz = xz[np.isfinite(xz)]

    # Data PSD
    f, S = mtm_psd(xz, fs=fs, NW=NW, Kmax=Kmax)

    # AR(1) parameters
    phi = ar1_phi(xz)
    var = np.var(xz)
    b = np.sqrt((1 - phi**2) * var)

    n = len(xz)
    spin = 200
    S_surr = np.zeros((nsurr, len(f)))

    for i in range(nsurr):
        y = np.zeros(n)
        state = 0.0
        white = rng.normal(0, 1, spin + n)
        for t in range(spin + n):
            state = phi * state + b * white[t]
            if t >= spin:
                y[t - spin] = state

        # standardize surrogate
        y = (y - y.mean()) / y.std(ddof=1)

        _, Sy = mtm_psd(y, fs=fs, NW=NW, Kmax=Kmax)
        S_surr[i, :] = Sy

    ci_dict = {p: np.percentile(S_surr, p, axis=0) for p in ci}
    return f, S, ci_dict, phi

def mtm_psd_ar1_ci_abs(x, fs=1.0, NW=2.0, Kmax=None, nsurr=2000, ci=(95,), seed=0):
    """
    Same as mtm_psd_ar1_ci, but uses absolute units (no z-score).
    Returns f, S, ci_dict, phi
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()  # keep mean removal

    # Data PSD
    f, S = mtm_psd(x, fs=fs, NW=NW, Kmax=Kmax)

    # AR(1) params (phi from lag-1 autocorr)
    # Use same definition but on de-meaned (not standardized)
    if len(x) < 3:
        raise ValueError("Time series too short.")
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    phi = float(np.clip(phi, -0.99, 0.99))

    # Innovation std so that AR(1) has same variance as x
    var = np.var(x)
    b = np.sqrt((1 - phi**2) * var)

    n = len(x)
    spin = 200
    S_surr = np.zeros((nsurr, len(f)))

    for i in range(nsurr):
        y = np.zeros(n)
        state = 0.0
        white = rng.normal(0, 1, spin + n)
        for t in range(spin + n):
            state = phi * state + b * white[t]
            if t >= spin:
                y[t - spin] = state

        y = y - y.mean()
        _, Sy = mtm_psd(y, fs=fs, NW=NW, Kmax=Kmax)
        S_surr[i, :] = Sy

    ci_dict = {p: np.percentile(S_surr, p, axis=0) for p in ci}
    return f, S, ci_dict, phi

def band_variance_from_psd(f, S, pmin=20, pmax=70):
    """
    Integrate PSD over the frequency band corresponding to periods [pmin, pmax] years.
    f: frequency [1/yr], one-sided (rfft) and includes f=0
    S: PSD values at f
    Returns band variance in the same variance units as the input time series.
    """
    f = np.asarray(f)
    S = np.asarray(S)

    # band in frequency space
    f_lo = 1.0 / pmax   # smaller freq (longer period)
    f_hi = 1.0 / pmin   # larger freq (shorter period)

    m = (f >= f_lo) & (f <= f_hi)
    if m.sum() < 2:
        return np.nan

    return np.trapezoid(S[m], f[m])  # integrate over frequency



#%% Read in data

#Choose which months you want (i.e. DJF or JJA)
month_start = 1
month_end   = 12

ts = 'yearly' #or 'montly' or 'yearly'

#fh      = netcdf.Dataset(directory_data+'TEMP_month_'+str(month_start)+'-'+str(month_end)+'_QE_year_0-2200.nc', 'r')

fh       = netcdf.Dataset(directory_data + 'TEMP_Atlantic_depth_averaged_100_300m_year_600-1500_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_forward      = fh.variables['time_month'][:]                     #Model years
    TEMP_forward       = fh.variables['TEMP_month'][:, :, 80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_forward      = fh.variables['time'][:]                     #Model years
    TEMP_forward       = fh.variables['TEMP'][:, :, 80:80+100]          #Sea level pressure (av\eraged over months) [hPa]

    
lon               = fh.variables['lon'][:, 80:80+100]            #Array of longitudes [degE]
lat               = fh.variables['lat'][:, 80:80+100]            #Array of latitudes [degN]
area              = fh.variables['area'][:, 80:80+100]            #Area of grid cells [m^2]

fh.close()

fh       = netcdf.Dataset(directory_data + 'TEMP_Atlantic_depth_averaged_100_300m_year_2900-3800_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_backward      = fh.variables['time_month'][:]                     #Model years
    #TEMP_backward       = fh.variables['TEMP_month'][:,220::,0:100]         #Sea level pressure (av\eraged over months) [hPa]
    TEMP_backward       = fh.variables['TEMP_month'][:, :, 80:80+100]
    
elif ts == 'yearly':
    time_backward      = fh.variables['time'][:]                     #Model years
    TEMP_backward       = fh.variables['TEMP'][:, :, 80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
fh.close()

fh       = netcdf.Dataset(directory_data + 'SST_Atlantic_year_600-1500_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_forward      = fh.variables['time_month'][:]                     #Model years
    SST_forward       = fh.variables['SST_month'][:, :, 80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_forward      = fh.variables['time'][:]                     #Model years
    SST_forward       = fh.variables['SST'][:, :, 80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
fh.close()

fh       = netcdf.Dataset(directory_data + 'SST_Atlantic_year_2900-3800_month_1-12_QE.nc', 'r')

if ts == 'monthly':
    time_backward      = fh.variables['time_month'][:]                     #Model years
    SST_backward       = fh.variables['SST_month'][:, :, 80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
    
elif ts == 'yearly':
    time_backward      = fh.variables['time'][:]                     #Model years
    SST_backward       = fh.variables['SST'][:, :, 80:80+100]         #Sea level pressure (av\eraged over months) [hPa]
fh.close()

plt.figure()
plt.contourf(lon, lat, TEMP_forward[0])

plt.figure()
plt.contourf(lon, lat, SST_forward[0])

#%%

lon_eof, lat_eof, eof_E1, time_E1, PC_E1_SST, VAR_E1_SST, EOF_E1_SST		= ReadinData(directory_data + 'EOF_AMV_SST_forward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_600_1500.nc')
lon_eof, lat_eof, eof_E2, time_E2, PC_E2_SST, VAR_E2_SST, EOF_E2_SST		= ReadinData(directory_data + 'EOF_AMV_SST_backward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_2900_3800.nc')

lon_eof, lat_eof, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory_data + 'EOF_AMV_TEMP_forward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_600_1500.nc')
lon_eof, lat_eof, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory_data + 'EOF_AMV_TEMP_backward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_2900_3800.nc')


#%%

lat_min = 0
lat_max = 60

if ts == 'monthly':
    lowpass = 10*12

if ts == 'yearly':
    lowpass = 10

amv_sst_forward, sst_mean_forward, sst_dt_forward, sst_lp_forward = compute_amv_index_eof_consistent(
    SST_forward, time_forward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=lowpass,
    detrend_order=2,
    remove_monthly_clim=True,
    standardize=False)

amv_sst_backward, sst_mean_backward, sst_dt_backward, sst_lp_backward = compute_amv_index_eof_consistent(
    SST_backward, time_backward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=lowpass,
    detrend_order=2,
    remove_monthly_clim=True,
    standardize=False)

amv_temp_forward, temp_mean_forward, temp_dt_forward, temp_lp_forward = compute_amv_index_eof_consistent(
    TEMP_forward, time_forward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=lowpass,
    detrend_order=2,
    remove_monthly_clim=True,
    standardize=False)

amv_temp_backward, temp_mean_backward, temp_dt_backward, temp_lp_backward = compute_amv_index_eof_consistent(
    TEMP_backward, time_backward, lat, lon, area,
    lat_min=lat_min, lat_max=lat_max,
    basin_mask_func=north_atlantic_mask,
    lowpass=lowpass,
    detrend_order=2,
    remove_monthly_clim=True,
    standardize=False)

amv_temp_forward_10_70, temp_mean_forward_10_70, temp_dt_forward_10_70, temp_lp_forward_10_70 = compute_amv_index_eof_consistent(
    TEMP_forward, time_forward, lat, lon, area,
    lat_min=10, lat_max=70,
    basin_mask_func=north_atlantic_mask,
    lowpass=lowpass,
    detrend_order=2,
    remove_monthly_clim=True,
    standardize=False)

amv_temp_backward_10_70, temp_mean_backward_10_70, temp_dt_backward_10_70, temp_lp_backward_10_70 = compute_amv_index_eof_consistent(
    TEMP_backward, time_backward, lat, lon, area,
    lat_min=10, lat_max=70,
    basin_mask_func=north_atlantic_mask,
    lowpass=lowpass,
    detrend_order=2,
    remove_monthly_clim=True,
    standardize=False)

#%%

plt.figure()

#plt.plot(temp_mean_forward, label='Basin-mean')
plt.plot(temp_dt_forward, label='Basin-mean DT')
plt.plot(temp_lp_forward, label='Basin-mean DT LP')
plt.plot(amv_temp_forward, '--', label='AMV')
plt.legend()

#%%

def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)

plt.figure()
plt.plot(time_forward, z(amv_sst_forward), label='EOF-like AMV')
#plt.plot(z(AMV_index_forward), label='AMV traditional')
plt.plot(time_E1, z(-PC_E1_SST[0,:]), label='PC1')
plt.legend()


#%% Correlation values

# t_pc: time axis for PC1 (after low-pass in EOF routine)
t_pc = time_E1              # whatever you stored as results["..."]["time"]
pc1  = PC_E1_SST[0, :]

# t_amv: time axis for amv_sst
t_amv = time_forward           # time axis you used when computing the basin-mean index
amv   = amv_sst_forward

# intersect times
t_common = np.intersect1d(t_pc, t_amv)

pc1_c  = pc1[np.isin(t_pc,  t_common)]
amv_c  = amv[np.isin(t_amv, t_common)]

r = np.corrcoef(z(-pc1_c), z(amv_c))[0,1]
print(r)

#%%

title_fs = 14
label_fs = 13
tick_fs = 12
legend_fs = 12

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
(ax1, ax2), (ax3, ax4) = axes

ax1.plot(time_forward, amv_temp_forward, color='darkorange', label='Temperature (100–300 m)')
ax1.plot(time_forward, amv_sst_forward, color='green', label='SST')
ax1.set_title(r'a) AMV indices - PI$^{\mathrm{on}}_{\mathrm{QE}}$', fontsize=title_fs)
ax1.set_xlabel('Time [model year]', fontsize=label_fs)
ax1.set_ylabel('AMV index [K]', fontsize=label_fs)
ax1.set_ylim(-0.3, 0.3)
ax1.grid()
ax1.legend(fontsize=legend_fs)
ax1.tick_params(axis='both', labelsize=tick_fs)

ax2.plot(time_backward, amv_temp_backward, color='darkorange', label='Temperature (100–300 m)')
ax2.plot(time_backward, amv_sst_backward, color='green', label='SST')
ax2.set_title(r'b) AMV indices - PI$^{\mathrm{off}}_{\mathrm{QE}}$', fontsize=title_fs)
ax2.set_xlabel('Time [model year]', fontsize=label_fs)
ax2.set_ylabel('AMV index [K]', fontsize=label_fs)
ax2.set_ylim(-0.3, 0.3)
ax2.grid()
ax2.legend(fontsize=legend_fs)
ax2.tick_params(axis='both', labelsize=tick_fs)

forward_95_interval_sst = np.percentile(amv_sst_forward, [2.5, 97.5])
backward_95_interval_sst = np.percentile(amv_sst_backward, [2.5, 97.5])

ax3.axvline(forward_95_interval_sst[0], color='orange', linestyle='--', linewidth=2, label='95 percentile')
ax3.axvline(forward_95_interval_sst[1], color='orange', linestyle='--', linewidth=2)
ax3.axvline(backward_95_interval_sst[0], color='blue', linestyle='--', linewidth=2, label='95 percentile')
ax3.axvline(backward_95_interval_sst[1], color='blue', linestyle='--', linewidth=2)

ax3.hist(amv_sst_forward, bins=100, color='orange', alpha=0.5, density=True,
         label=r'PI$^{\mathrm{on}}_{\mathrm{QE}}$')
ax3.hist(amv_sst_backward, bins=100, color='blue', alpha=0.5, density=True,
         label=r'PI$^{\mathrm{off}}_{\mathrm{QE}}$')

ax3.set_title('c) PDF of AMV index using SST data', fontsize=title_fs)
ax3.set_xlabel('AMV index [K]', fontsize=label_fs)
ax3.set_ylabel('PDF', fontsize=label_fs)
ax3.set_xlim(-0.3, 0.3)
ax3.set_ylim(0, 18)
ax3.grid()
ax3.legend(fontsize=legend_fs, loc='upper right')
ax3.tick_params(axis='both', labelsize=tick_fs)

forward_95_interval_temp = np.percentile(amv_temp_forward, [2.5, 97.5])
backward_95_interval_temp = np.percentile(amv_temp_backward, [2.5, 97.5])

ax4.axvline(forward_95_interval_temp[0], color='orange', linestyle='--', linewidth=2, label='95 percentile')
ax4.axvline(forward_95_interval_temp[1], color='orange', linestyle='--', linewidth=2)
ax4.axvline(backward_95_interval_temp[0], color='blue', linestyle='--', linewidth=2, label='95 percentile')
ax4.axvline(backward_95_interval_temp[1], color='blue', linestyle='--', linewidth=2)

ax4.hist(amv_temp_forward, bins=100, color='orange', alpha=0.5, density=True,
         label=r'PI$^{\mathrm{on}}_{\mathrm{QE}}$')
ax4.hist(amv_temp_backward, bins=100, color='blue', alpha=0.5, density=True,
         label=r'PI$^{\mathrm{off}}_{\mathrm{QE}}$')

ax4.set_title('d) PDF of AMV index using temperature (100–300 m) data', fontsize=title_fs)
ax4.set_xlabel('AMV index [K]', fontsize=label_fs)
ax4.set_ylabel('PDF', fontsize=label_fs)
ax4.set_xlim(-0.3, 0.3)
ax4.set_ylim(0, 18)
ax4.grid()
ax4.legend(fontsize=legend_fs)
ax4.tick_params(axis='both', labelsize=tick_fs)

plt.tight_layout()
plt.savefig(directory_figures + 'AMV_indices_and_PDFs_4panel.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%% Welch PSD

f, P, ci, a1 = Welch(temp_dt_forward, nperseg=256, nsurr=2000)

#Convert to period in years (avoid f=0)
mask = f > 0
period = 1 / f[mask]
Pp = P[mask]
ci95 = ci[95][mask]

nperseg   = 256      # good for ~900 years
nsurr     = 2000
ci_level  = 90
period_xlim = (2, 300)

f_on,  P_on,  ci_on,  a1_on  = Welch(temp_dt_forward, nperseg=nperseg, nsurr=nsurr, ci=(ci_level,), seed=1)
f_off, P_off, ci_off, a1_off = Welch(temp_dt_backward, nperseg=nperseg, nsurr=nsurr, ci=(ci_level,), seed=2)

mask_on  = f_on  > 0
mask_off = f_off > 0

per_on  = 1.0 / f_on[mask_on]
per_off = 1.0 / f_off[mask_off]

Pp_on  = P_on[mask_on]
Pp_off = P_off[mask_off]

CIp_on  = ci_on[ci_level][mask_on]
CIp_off = ci_off[ci_level][mask_off]

sort_on  = np.argsort(per_on)
sort_off = np.argsort(per_off)

per_on,  Pp_on,  CIp_on  = per_on[sort_on],  Pp_on[sort_on],  CIp_on[sort_on]
per_off, Pp_off, CIp_off = per_off[sort_off], Pp_off[sort_off], CIp_off[sort_off]

#plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

ax = axes[0]
ax.axvspan(20, 70, alpha=0.2, color='royalblue')
ax.plot(per_on, Pp_on, linewidth=1.8, color='royalblue', label="MTM PSD")
ax.plot(per_on, CIp_on, linestyle="--", color='red', linewidth=1.2, label=f"AR(1) {ci_level}%")
ax.set_xscale("log")
ax.set_xlim(*period_xlim)
ax.set_xlabel("Period [model years]")
ax.set_ylabel("Power [normalised]")
ax.set_title("a) PI$^{on}_{QE}$ - NA TEMP (Welch)")
#ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
#ax.text(0.02, 0.02, f"AR(1) ϕ={phi_on:.2f}", transform=ax.transAxes, va="bottom")
ax.grid(True, which="both", alpha=0.3)
#ax.legend(frameon=False, loc="upper right")

ax = axes[1]
ax.axvspan(20, 70, alpha=0.2, color='royalblue')
ax.plot(per_off, Pp_off, linewidth=1.8, color='royalblue', label="MTM PSD")
ax.plot(per_off, CIp_off, linestyle="--", linewidth=1.2, color='red', label=f"AR(1) {ci_level}%")
ax.set_xscale("log")
ax.set_xlim(*period_xlim)
ax.set_xlabel("Period [model years]")
ax.set_title("b) PI$^{off}_{QE}$ - NA TEMP (Welch)")
#ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, va="top", fontweight="bold")
#ax.text(0.02, 0.02, f"AR(1) ϕ={phi_off:.2f}", transform=ax.transAxes, va="bottom")
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
plt.show()

#%% multitaper, standardized 

#Settings
fs = 1.0
NW = 2.5
Kmax = None
nsurr = 10000
ci_level = 95
period_xlim = (2, 300)

#Compute MTM spectra + CI (use quadraticially detrended, annual mean, basin-mean data, NO LOWPASS FILTER!)

#SST
f_sst_on_stand,  S_sst_on_stand,  ci_sst_on_stand,  phi_sst_on_stand  = mtm_psd_ar1_ci(
    sst_lp_forward,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)
f_sst_off_stand, S_sst_off_stand, ci_sst_off_stand, phi_sst_off_stand = mtm_psd_ar1_ci(
    sst_lp_backward, fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)

#TEMP
f_tmp_on_stand,  S_tmp_on_stand,  ci_tmp_on_stand,  phi_tmp_on_stand  = mtm_psd_ar1_ci(
    temp_lp_forward,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)
f_tmp_off_stand, S_tmp_off_stand, ci_tmp_off_stand, phi_tmp_off_stand = mtm_psd_ar1_ci(
    temp_lp_backward, fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_sst_on_stand,  Sp_sst_on_stand,  CIp_sst_on_stand  = to_period_sorted(f_sst_on_stand,  S_sst_on_stand,  ci_sst_on_stand[ci_level])
per_sst_off_stand, Sp_sst_off_stand, CIp_sst_off_stand = to_period_sorted(f_sst_off_stand, S_sst_off_stand, ci_sst_off_stand[ci_level])

per_tmp_on_stand,  Sp_tmp_on_stand,  CIp_tmp_on_stand  = to_period_sorted(f_tmp_on_stand,  S_tmp_on_stand,  ci_tmp_on_stand[ci_level])
per_tmp_off_stand, Sp_tmp_off_stand, CIp_tmp_off_stand = to_period_sorted(f_tmp_off_stand, S_tmp_off_stand, ci_tmp_off_stand[ci_level])

#%%

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

panels = [
    (axes[0,0], per_sst_on_stand,  Sp_sst_on_stand,  CIp_sst_on_stand,  r"a) PI$^{on}_{QE}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)",  "a"),
    (axes[0,1], per_tmp_on_stand,  Sp_tmp_on_stand,  CIp_tmp_on_stand,  r"b) PI$^{on}_{QE}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)", "b"),
    (axes[1,0], per_sst_off_stand, Sp_sst_off_stand, CIp_sst_off_stand, r"c) PI$^{off}_{QE}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)", "c"),
    (axes[1,1], per_tmp_off_stand, Sp_tmp_off_stand, CIp_tmp_off_stand, r"d) PI$^{off}_{QE}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)","d"),
]

for ax, per, Sp, CIp, title, lab in panels:
    ax.axvspan(20, 70, alpha=0.2, color="royalblue")  # AMV band
    ax.plot(per, Sp, linewidth=1.8, color="royalblue", label="normalised MT spectrum")
    ax.plot(per, CIp, linestyle="--", linewidth=1.2, color="red", label=f"AR(1) {ci_level}%")

    ax.set_xscale("log")
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.3)

axes[0,0].set_ylabel("Power [normalised]", fontsize =11)
axes[1,0].set_ylabel("Power [normalised]", fontsize = 11)
axes[1,0].set_xlabel("Period [model years]", fontsize = 11)
axes[1,1].set_xlabel("Period [model years]", fontsize=11)

axes[0,0].legend(frameon=False, loc="upper left")

fig.tight_layout()
#plt.savefig(directory_figures + "MTM_2x2_NA_SST_TEMP_on_off.pdf", dpi=300, bbox_inches="tight")
plt.show()

#%% multitaper not standardized 

# --- compute absolute PSDs + envelopes ---
f_sst_on,  S_sst_on,  ci_sst_on,  phi_sst_on  = mtm_psd_ar1_ci_abs(sst_dt_forward,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)
f_sst_off, S_sst_off, ci_sst_off, phi_sst_off = mtm_psd_ar1_ci_abs(sst_dt_backward, fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)

f_tmp_on,  S_tmp_on,  ci_tmp_on,  phi_tmp_on  = mtm_psd_ar1_ci_abs(temp_dt_forward,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)
f_tmp_off, S_tmp_off, ci_tmp_off, phi_tmp_off = mtm_psd_ar1_ci_abs(temp_dt_backward, fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=1)

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_sst_on,  Sp_sst_on,  CIp_sst_on  = to_period_sorted(f_sst_on,  S_sst_on,  ci_sst_on[ci_level])
per_sst_off, Sp_sst_off, CIp_sst_off = to_period_sorted(f_sst_off, S_sst_off, ci_sst_off[ci_level])
per_tmp_on,  Sp_tmp_on,  CIp_tmp_on  = to_period_sorted(f_tmp_on,  S_tmp_on,  ci_tmp_on[ci_level])
per_tmp_off, Sp_tmp_off, CIp_tmp_off = to_period_sorted(f_tmp_off, S_tmp_off, ci_tmp_off[ci_level])


#%%

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

panels = [
    (axes[0,0], per_sst_on,  Sp_sst_on,  CIp_sst_on,  r"a) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[0,1], per_tmp_on,  Sp_tmp_on,  CIp_tmp_on,  r"b) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[1,0], per_sst_off, Sp_sst_off, CIp_sst_off, r"c) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[1,1], per_tmp_off, Sp_tmp_off, CIp_tmp_off, r"d) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
]

for ax, per, Sp, CIp, title in panels:
    ax.axvspan(20, 70, alpha=0.2, color="royalblue")
    ax.plot(per, Sp,  lw=1.8, color="royalblue", label="absolute MT spectrum")
    ax.plot(per, CIp, lw=1.2, color="red", ls="--", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    ax.set_yscale("log")   # absolute PSD often spans orders of magnitude
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

axes[0,0].set_ylabel("Power [$^\circ$C$^2$ / (1/yr)]", fontsize =11)
axes[1,0].set_ylabel("Power [$^\circ$C$^2$  / (1/yr)]", fontsize =11)
axes[1,0].set_xlabel("Period [model years]", fontsize =11)
axes[1,1].set_xlabel("Period [model years]", fontsize =11)

axes[0,0].legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.show()

#%% Absolute change in variance in MD band 

#SST
sst_band_on  = band_variance_from_psd(f_sst_on,  S_sst_on,  20, 70)
sst_band_off = band_variance_from_psd(f_sst_off, S_sst_off, 20, 70)

#TEMP band variance (absolute)
tmp_band_on  = band_variance_from_psd(f_tmp_on,  S_tmp_on,  20, 70)
tmp_band_off = band_variance_from_psd(f_tmp_off, S_tmp_off, 20, 70)

print("SST 20–70y band var:  on =", sst_band_on,  " off =", sst_band_off,  " ratio(off/on) =", sst_band_off/sst_band_on)
print("TMP 20–70y band var:  on =", tmp_band_on,  " off =", tmp_band_off,  " ratio(off/on) =", tmp_band_off/tmp_band_on)

#%%
def bandpower_fraction_from_psd(f, S, per_min=20, per_max=70):
    """
    Fraction of variance in a period band [per_min, per_max] (years),
    computed by integrating PSD S(f) over frequency.

    f: frequency array (cycles/year if fs=1)
    S: PSD array (units^2 per (cycles/year))
    """
    f = np.asarray(f, float)
    S = np.asarray(S, float)

    # keep positive freqs
    m = np.isfinite(f) & np.isfinite(S) & (f > 0)
    f = f[m]
    S = S[m]

    # sort by frequency (just in case)
    s = np.argsort(f)
    f = f[s]
    S = S[s]

    # period band -> frequency band
    f_low  = 1.0 / per_max   # 1/70
    f_high = 1.0 / per_min   # 1/20

    band = (f >= f_low) & (f <= f_high)

    total_var = np.trapezoid(S, f)
    band_var  = np.trapezoid(S[band], f[band])

    return band_var / total_var, band_var, total_var

frac_sst_on,  band_sst_on,  tot_sst_on  = bandpower_fraction_from_psd(f_sst_on,  S_sst_on,  20, 70)
frac_sst_off, band_sst_off, tot_sst_off = bandpower_fraction_from_psd(f_sst_off, S_sst_off, 20, 70)

print("SST fraction 20–70y (on): ", frac_sst_on)
print("SST fraction 20–70y (off):", frac_sst_off)
print("SST bandpower ratio off/on:", band_sst_off / band_sst_on)
print("SST total var ratio off/on:", tot_sst_off / tot_sst_on)

frac_tmp_on,  band_tmp_on,  tot_tmp_on  = bandpower_fraction_from_psd(f_tmp_on,  S_tmp_on,  20, 70)
frac_tmp_off, band_tmp_off, tot_tmp_off = bandpower_fraction_from_psd(f_tmp_off, S_tmp_off, 20, 70)

print("TEMP fraction 20–70y (on): ", frac_tmp_on)
print("TEMP fraction 20–70y (off):", frac_tmp_off)
print("TEMP bandpower ratio off/on:", band_tmp_off / band_tmp_on)
print("TEMP total var ratio off/on:", tot_tmp_off / tot_tmp_on)

sst_frac_on  = sst_band_on  / tot_sst_on
sst_frac_off = sst_band_off / tot_sst_off

print("SST band fraction on :", sst_frac_on)
print("SST band fraction off:", sst_frac_off)

temp_frac_on  = tmp_band_on  / tot_tmp_on
temp_frac_off = tmp_band_off / tot_tmp_off

print("TEMP band fraction on :", temp_frac_on)
print("TEMP band fraction off:", temp_frac_off)

#%%

# Values
labels = ['SST', 'TEMP']
band_on  = [sst_band_on,  tmp_band_on]
band_off = [sst_band_off, tmp_band_off]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6,4))

bars_on  = ax.bar(x - width/2, band_on,  width, label='AMOC on')
bars_off = ax.bar(x + width/2, band_off, width, label='AMOC off')

ax.set_ylabel('20–70 yr Band Variance [units²]')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('Multidecadal Variance (20–70 years)')
ax.legend(frameon=False)

# Optional: show ratio on top of bars
for i in range(len(labels)):
    ratio = band_off[i] / band_on[i]
    ax.text(x[i], max(band_on[i], band_off[i]) * 1.05,
            f'×{ratio:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=False, sharey=False)

# ----- Spectra panels (top 2 rows) -----
spectra_panels = [
    (axes[0,0], per_sst_on_stand,  Sp_sst_on_stand,  CIp_sst_on_stand,  r"a) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+"–"+str(lat_max)+r"$^\circ$N)", "royalblue"),
    (axes[0,1], per_tmp_on_stand,  Sp_tmp_on_stand,  CIp_tmp_on_stand,  r"b) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+"–"+str(lat_max)+r"$^\circ$N)", "royalblue"),
    (axes[1,0], per_sst_off_stand, Sp_sst_off_stand, CIp_sst_off_stand, r"c) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+"–"+str(lat_max)+r"$^\circ$N)", "indianred"),
    (axes[1,1], per_tmp_off_stand, Sp_tmp_off_stand, CIp_tmp_off_stand, r"d) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+"–"+str(lat_max)+r"$^\circ$N)", "indianred")]


for ax, per, Sp, CIp, title, col in spectra_panels:
    ax.axvspan(20, 70, alpha=0.15, color="grey")
    ax.plot(per, Sp,  lw=1.8, color=col, label="MT spectrum")
    ax.plot(per, CIp, lw=1.2, color="black", ls="--", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_ylim(0, 0.021)
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


# Share x among spectra only (per column)
axes[1,0].sharex(axes[0,0])
axes[1,1].sharex(axes[0,1])

# Labels for spectra
axes[0,0].set_ylabel(r"Power [normalised]", fontsize=11)
axes[1,0].set_ylabel(r"Power [normalised]", fontsize=11)
axes[0,1].set_ylabel(r"Power [normalised]", fontsize=11)
axes[1,1].set_ylabel(r"Power [normalised]", fontsize=11)

# Put x-labels only on row 1 (spectra bottom row)
axes[0,0].set_xlabel("Period [model years]", fontsize=11)
axes[0,1].set_xlabel("Period [model years]", fontsize=11)
axes[1,0].set_xlabel("Period [model years]", fontsize=11)
axes[1,1].set_xlabel("Period [model years]", fontsize=11)

axes[0,0].legend(frameon=False, loc="upper left")
axes[1,0].legend(frameon=False, loc="upper left")

# ----- Bar panels (bottom row) -----
ax_e = axes[2,0]
ax_f = axes[2,1]

# Panel (e): SST band variance
ax_e.set_title("e) 20–70 year band variance (NA SST)", fontsize=11)
obs_sst = np.array([sst_on_obs, sst_off_obs], float)
ci_sst_on  = pct_5_95(sst_on_surr)
ci_sst_off = pct_5_95(sst_off_surr)

x = np.arange(2)
ax_e.bar(x, obs_sst/1e-6, width=0.6, color=["royalblue", "indianred"])
ax_e.vlines(x[0], ci_sst_on[0],  ci_sst_on[1],  color="k", lw=1.5)
ax_e.vlines(x[1], ci_sst_off[0], ci_sst_off[1], color="k", lw=1.5)
ax_e.set_xticks(x)
ax_e.set_xticklabels(["PI$^{on}_{QE}$", "PI$^{off}_{QE}$"])
ax_e.set_ylabel(r"Variance [x 10$^{-6}$ $^\circ$C$^2$]", fontsize=11)
ax_e.set_xlim(-0.5, 1.5)

#ratio_sst = obs_sst[1] / obs_sst[0]
#ax_e.text(0.5, max(obs_sst)*1.05, f"×{ratio_sst:.2f}", ha="center", fontsize=10)

# Panel (f): TEMP band variance
ax_f.set_title("f) 20–70 year band variance (NA TEMP)", fontsize=11)
obs_tmp = np.array([tmp_on_obs, tmp_off_obs], float)
ci_tmp_on  = pct_5_95(tmp_on_surr)
ci_tmp_off = pct_5_95(tmp_off_surr)

ax_f.bar(x, obs_tmp/1e-6, width=0.6, color=["royalblue", "indianred"])
ax_f.vlines(x[0], ci_tmp_on[0],  ci_tmp_on[1],  color="k", lw=1.5)
ax_f.vlines(x[1], ci_tmp_off[0], ci_tmp_off[1], color="k", lw=1.5)
ax_f.set_xticks(x)
ax_f.set_xticklabels(["PI$^{on}_{QE}$", "PI$^{off}_{QE}$"])
ax_f.set_ylabel(r"Variance [x 10$^{-6}$ $^\circ$C$^2$]", fontsize=11)
#ax_f.set_xlabel("State", fontsize=11)
ax_f.set_xlim(-0.5, 1.5)

#ratio_tmp = obs_tmp[1] / obs_tmp[0]
#ax_f.text(0.5, max(obs_tmp)*1.05, f"×{ratio_tmp:.2f}", ha="center", fontsize=10)

fig.tight_layout()
plt.show()

#%%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

panels = [
    (axes[0,0], per_sst_on,  Sp_sst_on,  CIp_sst_on,  r"a) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[1,0], per_tmp_on,  Sp_tmp_on,  CIp_tmp_on,  r"c) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[0,1], per_sst_off, Sp_sst_off, CIp_sst_off, r"b) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[1,1], per_tmp_off, Sp_tmp_off, CIp_tmp_off, r"d) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
]

for ax, per, Sp, CIp, title in panels:
    ax.axvspan(20, 70, alpha=0.2, color="royalblue")
    ax.plot(per, Sp,  lw=1.8, color="royalblue", label="MT spectrum")
    ax.plot(per, CIp, lw=1.2, color="red", ls="--", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="major", alpha=0.3)

axes[0,0].set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
axes[1,0].set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
axes[1,0].set_xlabel("Period [model years]", fontsize=11)
axes[1,1].set_xlabel("Period [model years]", fontsize=11)

axes[0,0].legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.savefig(directory_figures + 'AMV_MTM_spectra_SST_TEMP.pdf')
plt.show()

#%%

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

panels = [
    (axes[0,0], per_sst_on_stand,  Sp_sst_on_stand,  CIp_sst_on_stand,  r"a) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[1,0], per_tmp_on_stand,  Sp_tmp_on_stand,  CIp_tmp_on_stand,  r"c) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[0,1], per_sst_off_stand, Sp_sst_off_stand, CIp_sst_off_stand, r"b) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA SST ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
    (axes[1,1], per_tmp_off_stand, Sp_tmp_off_stand, CIp_tmp_off_stand, r"d) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - NA TEMP ("+str(lat_min)+" - "+str(lat_max)+"$^\circ$N)"),
]

for ax, per, Sp, CIp, title in panels:
    ax.axvspan(20, 70, alpha=0.2, color="royalblue")
    ax.plot(per, Sp,  lw=1.8, color="royalblue", label="normalised MT spectrum")
    ax.plot(per, CIp, lw=1.2, color="red", ls="--", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.set_ylim(0,0.0008)
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="major", alpha=0.3)

axes[0,0].set_ylabel("Power [normalised]", fontsize=11)
axes[1,0].set_ylabel("Power [normalised]", fontsize=11)
axes[1,0].set_xlabel("Period [model years]", fontsize=11)
axes[1,1].set_xlabel("Period [model years]", fontsize=11)

axes[0,0].legend(frameon=False, loc="upper right")

fig.tight_layout()
#plt.savefig(directory_figures + 'AMV_MTM_spectra_SST_TEMP_normalised_LP_monthly.pdf')
plt.show()

#%%

def band_variance_from_psd(f, S, pmin=20, pmax=70):
    """
    Integrate one-sided PSD S(f) over the frequency band corresponding to periods [pmin, pmax] years.
    f: [1/yr], includes 0; S same length.
    Returns variance contribution in that band (units^2).
    """
    f = np.asarray(f); S = np.asarray(S)

    flo = 1.0 / pmax
    fhi = 1.0 / pmin
    m = (f >= flo) & (f <= fhi)

    if m.sum() < 2:
        return np.nan
    return np.trapezoid(S[m], f[m])

from numpy.random import default_rng

def ar1_phi_from_series(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    if len(x) < 3:
        return np.nan
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    return float(np.clip(phi, -0.99, 0.99))

def mtm_band_mc_ar1_abs(x, fs=1.0, NW=2.5, Kmax=None,
                        pmin=20, pmax=70, nsurr=5000, seed=0):
    """
    Absolute-units MTM:
      - compute observed band-integrated variance (20–70y by default)
      - compute AR(1) surrogate distribution of band variance
    Returns:
      band_obs, band_surr (array), phi
    """
    rng = default_rng(seed)

    # Prepare observed series (absolute)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()

    # Observed PSD + band variance
    f_obs, S_obs = mtm_psd(x, fs=fs, NW=NW, Kmax=Kmax)
    band_obs = band_variance_from_psd(f_obs, S_obs, pmin=pmin, pmax=pmax)

    # AR(1) parameters matched to x
    phi = ar1_phi_from_series(x)
    var = np.var(x)
    b = np.sqrt((1 - phi**2) * var)

    # Monte Carlo: store only band variance (memory-light)
    n = len(x)
    spin = 200
    band_surr = np.empty(nsurr, float)

    for i in range(nsurr):
        y = np.zeros(n)
        state = 0.0
        white = rng.normal(0, 1, spin + n)

        for t in range(spin + n):
            state = phi * state + b * white[t]
            if t >= spin:
                y[t - spin] = state

        y = y - y.mean()
        f_y, S_y = mtm_psd(y, fs=fs, NW=NW, Kmax=Kmax)
        band_surr[i] = band_variance_from_psd(f_y, S_y, pmin=pmin, pmax=pmax)

    return band_obs, band_surr, phi

# Settings
fs = 1.0
NW = 2.5
Kmax = None
pmin, pmax = 20, 70
nsurr = 5000  # you asked for MC; 5000 is often enough, 10000 is fine if it runs fast enough

# Compute band variance + surrogate distributions
sst_on_obs,  sst_on_surr,  phi_sst_on  = mtm_band_mc_ar1_abs(sst_dt_forward,  fs=fs, NW=NW, Kmax=Kmax, pmin=pmin, pmax=pmax, nsurr=nsurr, seed=1)
sst_off_obs, sst_off_surr, phi_sst_off = mtm_band_mc_ar1_abs(sst_dt_backward, fs=fs, NW=NW, Kmax=Kmax, pmin=pmin, pmax=pmax, nsurr=nsurr, seed=2)

tmp_on_obs,  tmp_on_surr,  phi_tmp_on  = mtm_band_mc_ar1_abs(temp_dt_forward,  fs=fs, NW=NW, Kmax=Kmax, pmin=pmin, pmax=pmax, nsurr=nsurr, seed=3)
tmp_off_obs, tmp_off_surr, phi_tmp_off = mtm_band_mc_ar1_abs(temp_dt_backward, fs=fs, NW=NW, Kmax=Kmax, pmin=pmin, pmax=pmax, nsurr=nsurr, seed=4)

# One-sided p-value: P(surrogate >= observed)
def pval_upper(obs, surr):
    return (np.sum(surr >= obs) + 1) / (len(surr) + 1)

p_sst_on  = pval_upper(sst_on_obs,  sst_on_surr)
p_sst_off = pval_upper(sst_off_obs, sst_off_surr)
p_tmp_on  = pval_upper(tmp_on_obs,  tmp_on_surr)
p_tmp_off = pval_upper(tmp_off_obs, tmp_off_surr)

print("p-values (upper-tail vs AR1):")
print("SST on :", p_sst_on,  "phi:", phi_sst_on)
print("SST off:", p_sst_off, "phi:", phi_sst_off)
print("TMP on :", p_tmp_on,  "phi:", phi_tmp_on)
print("TMP off:", p_tmp_off, "phi:", phi_tmp_off)

#%% multitaper not standardized  for PC1 

# --- compute absolute PSDs + envelopes ---
f_sst_on,  S_sst_on,  ci_sst_on,  phi_sst_on  = mtm_psd_ar1_ci_abs(PC_E1_SST[0,:],  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=11)
f_sst_off, S_sst_off, ci_sst_off, phi_sst_off = mtm_psd_ar1_ci_abs(PC_E2_SST[0,:], fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=12)

f_tmp_on,  S_tmp_on,  ci_tmp_on,  phi_tmp_on  = mtm_psd_ar1_ci_abs(PC_E1[0,:],  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=13)
f_tmp_off, S_tmp_off, ci_tmp_off, phi_tmp_off = mtm_psd_ar1_ci_abs(PC_E2[0,:], fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=14)

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_sst_on,  Sp_sst_on,  CIp_sst_on  = to_period_sorted(f_sst_on,  S_sst_on,  ci_sst_on[ci_level])
per_sst_off, Sp_sst_off, CIp_sst_off = to_period_sorted(f_sst_off, S_sst_off, ci_sst_off[ci_level])
per_tmp_on,  Sp_tmp_on,  CIp_tmp_on  = to_period_sorted(f_tmp_on,  S_tmp_on,  ci_tmp_on[ci_level])
per_tmp_off, Sp_tmp_off, CIp_tmp_off = to_period_sorted(f_tmp_off, S_tmp_off, ci_tmp_off[ci_level])

#%%

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

panels = [
    (axes[0,0], per_sst_on,  Sp_sst_on,  CIp_sst_on,  r"a) PI$^{on}_{QE}$ - NA SST PC1"),
    (axes[1,0], per_tmp_on,  Sp_tmp_on,  CIp_tmp_on,  r"c) PI$^{on}_{QE}$ - NA TEMP PC1"),
    (axes[0,1], per_sst_off, Sp_sst_off, CIp_sst_off, r"b) PI$^{off}_{QE}$ - NA SST PC1"),
    (axes[1,1], per_tmp_off, Sp_tmp_off, CIp_tmp_off, r"d) PI$^{off}_{QE}$ - NA TEMP PC1"),
]

for ax, per, Sp, CIp, title in panels:
    ax.axvspan(20, 70, alpha=0.2, color="royalblue")
    ax.plot(per, Sp,  lw=1.8, color="royalblue", label="MT spectrum")
    ax.plot(per, CIp, lw=1.2, color="red", ls="--", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="major", alpha=0.3)

axes[0,0].set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
axes[1,0].set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
axes[1,0].set_xlabel("Period [model years]", fontsize=11)
axes[1,1].set_xlabel("Period [model years]", fontsize=11)

axes[0,0].legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.savefig(directory_figures + 'AMV_MTM_spectra_SST_TEMP_PC1.pdf')
plt.show()

#%% MTM for amoc strength

# --- compute absolute PSDs + envelopes ---
f_amoc_on,  S_amoc_on,  ci_amoc_on,  phi_amoc_on  = mtm_psd_ar1_ci_abs(AMOC_dt_forward,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=11)
f_amoc_off, S_amoc_off, ci_amoc_off, phi_amoc_off = mtm_psd_ar1_ci_abs(AMOC_dt_backward, fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=12)

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_amoc_on,  Sp_amoc_on,  CIp_amoc_on  = to_period_sorted(f_amoc_on,  S_amoc_on,  ci_amoc_on[ci_level])
per_amoc_off, Sp_amoc_off, CIp_amoc_off = to_period_sorted(f_amoc_off, S_amoc_off, ci_amoc_off[ci_level])

#%%

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

panels = [
    (axes[0,0], per_amoc_on,  Sp_amoc_on,  CIp_amoc_on,  r"a) PI$^{on}_{QE}$ - AMOC (26N)"),
    (axes[1,0], per_tmp_on,  Sp_tmp_on,  CIp_tmp_on,  r"c) PI$^{on}_{QE}$ - NA TEMP PC1"),
    (axes[0,1], per_amoc_off, Sp_amoc_off, CIp_amoc_off, r"b) PI$^{off}_{QE}$ - AMOC (26N)"),
    (axes[1,1], per_tmp_off, Sp_tmp_off, CIp_tmp_off, r"d) PI$^{off}_{QE}$ - NA TEMP PC1"),
]

for ax, per, Sp, CIp, title in panels:
    ax.axvspan(20, 70, alpha=0.2, color="royalblue")
    ax.plot(per, Sp,  lw=1.8, color="royalblue", label="MT spectrum")
    ax.plot(per, CIp, lw=1.2, color="red", ls="--", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="major", alpha=0.3)

axes[0,0].set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
axes[1,0].set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
axes[1,0].set_xlabel("Period [model years]", fontsize=11)
axes[1,1].set_xlabel("Period [model years]", fontsize=11)

axes[0,0].legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.savefig(directory_figures + 'MTM_spectra_AMOC.pdf')
plt.show()

