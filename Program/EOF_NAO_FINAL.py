#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:54:57 2026

@author: 6008399
"""

#%%
import numpy as np
import numpy.ma as ma
from numpy.linalg import svd

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
import matplotlib.colors as mcolors
import cartopy.mpl.ticker as cticker
import numpy as np
import numpy.ma as ma

#Making pathway to folder with all data
#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%%
#Choose which months you want (i.e. DJF or JJA)
month_start = 12
month_end   = 14

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch600_year_999-1100.nc', 'r')

time_month_E1       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_month_E1        = fh.variables['SLP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch1500_year_1899-2000.nc', 'r')

time_month_E2       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_month_E2        = fh.variables['SLP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch2900_year_2900-3500.nc', 'r')

time_month_E3       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_month_E3        = fh.variables['SLP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch3800_year_4199-4300.nc', 'r')

time_month_E4       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_month_E4        = fh.variables['SLP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

#Read in data
fh      = netcdf.Dataset(directory_data+'Atmosphere_DX_DY_AREA.nc', 'r')

dy      = fh.variables['DY'][:] #Grid spacing in y-direction
dx      = fh.variables['DX'][:] #Grid spacing in x-direction
area    = fh.variables['AREA'][:]

fh.close()

#%% Take only january (month = 1)

# month_start = 1
# month_end   = 1

# time_jan_E1 = time_month_E1[::12]
# time_jan_E2 = time_month_E2[::12]
# time_jan_E3 = time_month_E3[::12]
# time_jan_E4 = time_month_E4[::12]

# SLP_jan_E1 = SLP_month_E1[::12]
# SLP_jan_E2 = SLP_month_E2[::12]
# SLP_jan_E3 = SLP_month_E3[::12]
# SLP_jan_E4 = SLP_month_E4[::12]

plt.figure()
plt.contourf(lon, lat, SLP_month_E1[0])

#%%

def remove_monthly_climatology(x):
    """
    x: (ntime, nspace) anomalies matrix (can be masked)
    Assumes x is monthly and starts at January.
    Removes the mean seasonal cycle (12-month climatology) per gridpoint.
    """
    x = x.copy()
    ntime, nspace = x.shape
    for m in range(12):
        idx = np.arange(m, ntime, 12)
        clim = x[idx, :].mean(axis=0)
        x[idx, :] = x[idx, :] - clim
    return x

def detrend_poly(time, x, order=1):
    """
    Detrend each column of x with a polynomial of given order.
    time: (ntime,)
    x: (ntime, nspace) masked array or ndarray
    """
    x = ma.array(x, copy=True)
    t = np.asarray(time)

    for j in range(x.shape[1]):
        col = x[:, j]
        valid = ~ma.getmaskarray(col)
        if valid.sum() < max(10, order + 2):
            x[:, j] = ma.masked  # too few points
            continue

        y = col[valid].filled(np.nan)
        tt = t[valid]

        # Guard against constant series
        if np.nanstd(y) < 1e-12:
            x[:, j] = ma.masked
            continue

        p = np.polyfit(tt, y, order)
        fit = np.polyval(p, t)
        # subtract fit only where valid; preserve mask
        x[:, j] = ma.array(col - fit, mask=ma.getmaskarray(col))

    return x

def perform_eof_analysis(
    SST, time, lat, lon, area,
    remove_month=True,
    detrend_order=0,              # 0 = no detrend, 1 = linear, 2 = quadratic, ...
    lowpass=None,                 # None, or dict like {"method":"runmean","window":120}
    standardize=False,            # False = covariance EOFs (typical), True = correlation EOFs
    neof=5
):
    """
    SST: (ntime, nlat, nlon) masked array preferred
    lat/lon: 2D arrays (nlat,nlon) or compatible with plotting
    area: (nlat,nlon) grid cell area (masked or not)

    Returns:
      EOFs: (neof, nlat, nlon) spatial patterns (unweighted)
      PCs:  (neof, ntime_out) principal components (amplitude time series)
      var_frac: (neof,) explained variance fraction (0-1)
      svals: (nmode,) singular values
      time_out: corresponding time axis
    """

    SST = ma.array(SST, copy=False)
    ntime, nlat, nlon = SST.shape

    # Define spatial mask from the first timestep
    mask2d = ma.getmaskarray(SST[0, :, :])

    # Flatten valid ocean points
    valid_flat = ~mask2d.ravel()
    nspace = valid_flat.sum()
    
    plt.figure()
    plt.contourf(SST[0])

    # Build X as (ntime, nspace)
    X = ma.masked_all((ntime, nspace))
    SST_flat = SST.reshape(ntime, nlat*nlon)
    X[:, :] = SST_flat[:, valid_flat]

    # Remove monthly climatology
    if remove_month:
        X = remove_monthly_climatology(X)

    # Detrend
    if detrend_order and detrend_order > 0:
        X = detrend_poly(time, X, order=detrend_order)

    # Demean (temporal mean per gridpoint)
    X = X - X.mean(axis=0)
    
    print("X shape:", X.shape)
    print("X finite frac:", float(np.isfinite(X.filled(np.nan)).mean()))
    print("X masked frac:", float(ma.getmaskarray(X).mean()))
    print(np.nanmean(X))

    # Optional standardization (correlation EOFs)
    if standardize:
        std = X.std(axis=0)
        # mask near-zero std
        bad = std < 1e-12
        X[:, bad] = ma.masked
        std = ma.masked_array(std, mask=bad)
        X = X / std

    # Optional low-pass
    time_out = np.asarray(time)
    if lowpass is not None:
        method = lowpass.get("method", "runmean")
        if method == "runmean":
            window = int(lowpass.get("window", 1))
            X_lp = lowpass_running_mean(X, window=window)
            # center time for running mean
            start = (window - 1)//2
            time_out = time_out[start:start + X_lp.shape[0]]
            X = X_lp
        elif method == "butter":
            # NOTE: requires fully valid columns; masked cols stay masked
            X = lowpass_butterworth(
                X,
                fs_per_year=float(lowpass.get("fs_per_year", 12.0)),
                cutoff_years=float(lowpass.get("cutoff_years", 10.0)),
                order=int(lowpass.get("order", 4))
            )
        else:
            raise ValueError("lowpass method must be 'runmean' or 'butter'")

    # Area weights (sqrt(area)) applied to X columns
    area2d = ma.array(area, copy=False)
    w2d = ma.sqrt(area2d)
    w_flat = w2d.ravel()[valid_flat]

    # normalize weights (optional, but keeps magnitudes reasonable)
    w_flat = w_flat / ma.mean(w_flat)

    Xw = X * w_flat  # broadcast over time
    
    Xw = X * w_flat  # broadcast over time

    # Drop any columns that became masked (e.g., constant/invalid)
    colmask = ma.getmaskarray(Xw).any(axis=0)
    Xw2 = Xw[:, ~colmask]
    w2  = w_flat[~colmask]

    # Fill remaining masks with 0 (safe because we removed fully-masked columns)
    Xw2_filled = Xw2.filled(0.0)
    
    # SVD on (ntime, nspace_eff)
    # Xw = U S Vt ; EOFs live in Vt (spatial), PCs are U*S
    U, s, Vt = svd(Xw2_filled, full_matrices=False)
    
    print(U, s, Vt)

    # Explained variance fraction
    eig = s**2
    var_frac = eig / eig.sum()

    ne = min(neof, Vt.shape[0])

    # PCs (amplitude time series)
    PCs = (U[:, :ne] * s[:ne])  # (ntime_out, ne)
    PCs = PCs.T                 # (ne, ntime_out)
    
    print(PCs)

    # Spatial EOFs in *unweighted* space:
    # Vt gives patterns in weighted space; divide by weights to return to physical grid scaling
    EOF_space = (Vt[:ne, :].T / w2[:, None]).T # (ne, nspace_eff)

    # Map EOFs back onto full grid
    EOFs = ma.masked_all((ne, nlat*nlon))
    # put patterns back into the valid_flat subset, but only where we kept columns
    valid_idx = np.where(valid_flat)[0]
    kept_idx = valid_idx[~colmask]
    for k in range(ne):
        EOFs[k, kept_idx] = EOF_space[k, :]

    EOFs = EOFs.reshape(ne, nlat, nlon)

    return EOFs, PCs, var_frac[:ne], s, time_out

# Process all SLP datasets
lon1, lon2 = -90, 40
lat1, lat2 = 20, 80
#time1, time2 = 0, 2200
#depth_level = 500

lat_min_index  = (np.abs(lat - lat1)).argmin()
lat_max_index  = (np.abs(lat - lat2)).argmin()+1
lon_min_index   = (np.abs(lon - lon1)).argmin()
lon_max_index   = (np.abs(lon - lon2)).argmin()+1

datasets = {
    "SLP_E1": (SLP_month_E1[0:101,lat_min_index:lat_max_index, lon_min_index:lon_max_index], time_month_E1[0:101]),
    "SLP_E2": (SLP_month_E2[0:101,lat_min_index:lat_max_index, lon_min_index:lon_max_index], time_month_E2[0:101]),
    "SLP_E3": (SLP_month_E3[399:500,lat_min_index:lat_max_index, lon_min_index:lon_max_index], time_month_E3[399:500]),
    "SLP_E4": (SLP_month_E4[0:101,lat_min_index:lat_max_index, lon_min_index:lon_max_index], time_month_E4[0:101])}
    
#datasets = {
#    "TEMP_forward": (TEMP_month_forward, time_month_forward), 
#    "TEMP_backward": (TEMP_month_backward, time_month_backward)}

results = {}

for name, (SST, time) in datasets.items():
    print(f"Processing {name}...")
    print(np.shape(SST), np.shape(time), np.shape(lat), np.shape(lon), np.shape(area))

    EOFs, PCs, var_frac, svals, time_lp = perform_eof_analysis(
        SST, time, lat[lat_min_index:lat_max_index], lon[lon_min_index:lon_max_index], area[lat_min_index:lat_max_index, lon_min_index:lon_max_index],
        remove_month=False,
        detrend_order=1,
        lowpass=None,
        #lowpass = None,
        standardize=False,
        neof=5
    )

    results[name] = {
        "EOF": EOFs,           # (neof, nlat, nlon)
        "PC": PCs,             # (neof, ntime_lp)
        "var_frac": var_frac,  # (neof,) fraction (0-1)
        "svals": svals,        # singular values (all modes)
        "time": time_lp
    }

    print(f"Finished processing {name}.\n")

    # ---- Save to NetCDF ----
    neof = EOFs.shape[0]
    ntime_lp = len(time_lp)

    filename = (
        f"{directory_data}EOF_NAO_{name}_month_{month_start}_{month_end}"
        f"_detrend1_CESM_year_{int(time[0])}_{int(time[-1])}.nc")
    #filename = (
    #    f"{directory_data}EOF_AMV_{name}_month_{month_start}_{month_end}"
    #    f"_lowpass_none_detrend1_CESM_QE_year_{int(time[0])}_{int(time[-1])}.nc")
    print(f"Saving results to {filename}...")

    fh = netcdf.Dataset(filename, "w")

    fh.createDimension("lat", len(lat[lat_min_index:lat_max_index]))
    fh.createDimension("lon", len(lon[lon_min_index:lon_max_index]))
    fh.createDimension("eof", neof)
    fh.createDimension("time", ntime_lp)

    vlat = fh.createVariable("lat", "f4", ("lat"), zlib=True)
    vlon = fh.createVariable("lon", "f4", ("lon"), zlib=True)
    vtime = fh.createVariable("time", "f8", ("time",), zlib=True)
    veofn = fh.createVariable("eof", "i4", ("eof",), zlib=True)

    vEOF = fh.createVariable("EOF", "f4", ("eof", "lat", "lon"), zlib=True)
    vPC  = fh.createVariable("PC",  "f4", ("eof", "time"), zlib=True)
    vVAR = fh.createVariable("VAR", "f4", ("eof",), zlib=True)

    vlon.longname = "Array of longitudes"
    vlat.longname = "Array of latitudes"
    vtime.units = "Model year"
    vlon.units = "Degrees east"
    vlat.units = "Degrees north"
    vPC.long_name = "Principal components (amplitude time series)"
    vEOF.long_name = "EOF spatial patterns (unweighted)"
    vVAR.long_name = "Explained variance fraction"
    vVAR.units = "1"  # fraction; multiply by 100 in post if you want %

    # Write data
    vlon[:] = lon[lon_min_index:lon_max_index]
    vlat[:] = lat[lat_min_index:lat_max_index]
    vtime[:] = time_lp
    veofn[:] = np.arange(1, neof + 1)

    vEOF[:] = EOFs
    vPC[:]  = PCs
    vVAR[:] = var_frac  # already fraction (0-1)

    fh.close()
    print(f"Results saved to {filename}.\n")
    

# %%
