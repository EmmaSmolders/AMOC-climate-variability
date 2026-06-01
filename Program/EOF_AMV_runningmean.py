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
directory_data		= '/home/smolders/CESM_Collapse/Data/CESM/Ocean/'

#%%

#Choose which months you want (i.e. DJF or JJA)
month_start = 1
month_end   = 12

#Use either depth-averaged temepartures or SSTs for the EOF analysis
fh      = netcdf.Dataset(directory_data+'TEMP_Atlantic_depth_averaged_100_300m_year_600-1500_month_1-12_QE.nc', 'r')

time_month_forward      = fh.variables['time_month'][:]     #Model months
lon                     = fh.variables['lon'][220::,:]             #Array of longitudes [degE]
lat                     = fh.variables['lat'][220::,:]             #Array of latitudes [degN]
area                    = fh.variables['area'][220::,:] 
TEMP_month_forward       = fh.variables['TEMP_month'][:,220::,:]       #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'TEMP_Atlantic_depth_averaged_100_300m_year_2900-3800_month_1-12_QE.nc', 'r')

time_month_backward      = fh.variables['time_month'][:]     #Model months
#lon                     = fh.variables['lon'][220::,0:100]             #Array of longitudes [degE]
#lat                     = fh.variables['lat'][220::,0:100]             #Array of latitudes [degN]
#area                    = fh.variables['area'][220::,0:100] 
TEMP_month_backward       = fh.variables['TEMP_month'][:,220::,:]        #Sea level pressure (av\eraged over months) [hPa]

fh.close()

#print(time_month_backward[:24])

#sys.exit()
#-------------------------------------------------------------------------------------------------------------------------------------

#fh       = netcdf.Dataset(directory_data + 'SST_Atlantic_year_600-1500_month_1-12_QE.nc', 'r')

#time_month_forward      = fh.variables['time_month'][:]     #Model months
#lon                     = fh.variables['lon'][220::,:]             #Array of longitudes [degE]
#lat                     = fh.variables['lat'][220::,:]             #Array of latitudes [degN]
#area                    = fh.variables['area'][220::,:] 
#SST_month_forward       = fh.variables['SST_month'][:,220::,:]       #Sea level pressure (av\eraged over months) [hPa]

#fh.close()

#fh       = netcdf.Dataset(directory_data + 'SST_Atlantic_year_2900-3800_month_1-12_QE.nc', 'r')

#time_month_backward      = fh.variables['time_month'][:]     #Model months
#lon                     = fh.variables['lon'][220::,0:100]             #Array of longitudes [degE]
#lat                     = fh.variables['lat'][220::,0:100]             #Array of latitudes [degN]
#area                    = fh.variables['area'][220::,0:100] 
#SST_month_backward       = fh.variables['SST_month'][:,220::,:]       #Sea level pressure (av\eraged over months) [hPa]

#fh.close()

plt.figure()
plt.contourf(lon, lat, TEMP_month_forward[0,:,:])
plt.show()

mask70 = lat > 70                      #shape (164, 100)
mask70_3d = np.broadcast_to(mask70, TEMP_month_forward.shape)

TEMP_month_forward_70 = ma.masked_where(mask70_3d, TEMP_month_forward)
TEMP_month_backward_70 = ma.masked_where(np.broadcast_to(mask70, TEMP_month_backward.shape), TEMP_month_backward)

area_70 = ma.masked_where(mask70, area)

plt.figure()
plt.contourf(lon, lat, TEMP_month_forward[0,:,:])
plt.show()

plt.figure()
plt.contourf(lon, lat, TEMP_month_forward_70[0,:,:])
plt.ylim(0,70)
plt.show()

#sys.exit()

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

def lowpass_running_mean(x, window):
    """
    Simple low-pass via centered running mean.
    x: (ntime, nspace) masked array
    Returns shortened series: ntime-window+1
    """
    x = ma.array(x, copy=False)
    if window <= 1:
        return x

    ntime, nspace = x.shape
    out = ma.masked_all((ntime - window + 1, nspace))
    for i in range(ntime - window + 1):
        out[i, :] = x[i:i+window, :].mean(axis=0)
    return out

def lowpass_butterworth(x, fs_per_year=12.0, cutoff_years=10.0, order=4):
    """
    Butterworth low-pass (zero-phase). Requires scipy.
    cutoff_years=10 means pass periods longer than ~10 years.
    """
    from scipy.signal import butter, filtfilt

    x = ma.array(x, copy=True)
    ntime, nspace = x.shape

    # cutoff frequency in cycles per month:
    # cutoff in cycles/year = 1/cutoff_years
    # normalized cutoff for digital filter uses Nyquist (fs/2)
    cutoff_cy_per_year = 1.0 / cutoff_years
    nyq = fs_per_year / 2.0
    Wn = cutoff_cy_per_year / nyq  # normalized

    b, a = butter(order, Wn, btype="low")

    # apply per column on valid segments (simple version: require fully valid)
    out = ma.masked_all_like(x)
    for j in range(nspace):
        col = x[:, j]
        if ma.getmaskarray(col).any():
            continue  # keep masked; or implement segment-wise filtering
        y = col.filled(np.nan)
        if np.nanstd(y) < 1e-12:
            continue
        out[:, j] = filtfilt(b, a, y)
    return out

def perform_eof_analysis(
    TEMP, time, lat, lon, area,
    remove_month=True,
    detrend_order=0,              # 0 = no detrend, 1 = linear, 2 = quadratic, ...
    lowpass=None,                 # None, or dict like {"method":"runmean","window":120}
    standardize=False,            # False = covariance EOFs (typical), True = correlation EOFs
    neof=5
):
    """
    TEMP: (ntime, nlat, nlon) masked array preferred
    lat/lon: 2D arrays (nlat,nlon) or compatible with plotting
    area: (nlat,nlon) grid cell area (masked or not)

    Returns:
      EOFs: (neof, nlat, nlon) spatial patterns (unweighted)
      PCs:  (neof, ntime_out) principal components (amplitude time series)
      var_frac: (neof,) explained variance fraction (0-1)
      svals: (nmode,) singular values
      time_out: corresponding time axis
    """

    TEMP = ma.array(TEMP, copy=False)
    ntime, nlat, nlon = TEMP.shape

    # Define spatial mask from the first timestep
    mask2d = ma.getmaskarray(TEMP[0, :, :])

    # Flatten valid ocean points
    valid_flat = ~mask2d.ravel()
    nspace = valid_flat.sum()

    # Build X as (ntime, nspace)
    X = ma.masked_all((ntime, nspace))
    TEMP_flat = TEMP.reshape(ntime, nlat*nlon)
    X[:, :] = TEMP_flat[:, valid_flat]

    # Remove monthly climatology
    if remove_month:
        X = remove_monthly_climatology(X)

    # Detrend
    if detrend_order and detrend_order > 0:
        X = detrend_poly(time, X, order=detrend_order)

    # Demean (temporal mean per gridpoint)
    X = X - X.mean(axis=0)

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

    # Drop any columns that became masked (e.g., constant/invalid)
    colmask = ma.getmaskarray(Xw).any(axis=0)
    Xw2 = Xw[:, ~colmask]
    w2  = w_flat[~colmask]

    # Fill remaining masks with 0 (safe because we removed fully-masked columns)
    Xw2_filled = Xw2.filled(0.0)

    # SVD on (ntime, nspace_eff)
    # Xw = U S Vt ; EOFs live in Vt (spatial), PCs are U*S
    U, s, Vt = svd(Xw2_filled, full_matrices=False)

    # Explained variance fraction
    eig = s**2
    var_frac = eig / eig.sum()

    ne = min(neof, Vt.shape[0])

    # PCs (amplitude time series)
    PCs = (U[:, :ne] * s[:ne])  # (ntime_out, ne)
    PCs = PCs.T                 # (ne, ntime_out)

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
    
datasets = {
    "TEMP_forward": (TEMP_month_forward_70, time_month_forward), 
    "TEMP_backward": (TEMP_month_backward_70, time_month_backward)}
    
#datasets = {
#    "TEMP_forward": (TEMP_month_forward, time_month_forward), 
#    "TEMP_backward": (TEMP_month_backward, time_month_backward)}

results = {}

for name, (TEMP, time) in datasets.items():
    print(f"Processing {name}...")
    print(np.shape(TEMP), np.shape(time), np.shape(lat), np.shape(lon), np.shape(area))

    EOFs, PCs, var_frac, svals, time_lp = perform_eof_analysis(
        TEMP, time, lat, lon, area,
        remove_month=True,
        detrend_order=2,
        lowpass={"method": "runmean", "window": 120},
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
        f"{directory_data}EOF_AMV_{name}_month_{month_start}_{month_end}"
        f"_lowpass_runmean_120mo_detrend2_CESM_QE_year_{int(time[0])}_{int(time[-1])}.nc")
    #filename = (
    #    f"{directory_data}EOF_AMV_{name}_month_{month_start}_{month_end}"
    #    f"_lowpass_none_detrend1_CESM_QE_year_{int(time[0])}_{int(time[-1])}.nc")
    print(f"Saving results to {filename}...")

    fh = netcdf.Dataset(filename, "w")

    fh.createDimension("lat", lat.shape[0])
    fh.createDimension("lon", lon.shape[1])
    fh.createDimension("eof", neof)
    fh.createDimension("time", ntime_lp)

    vlat = fh.createVariable("lat", "f4", ("lat", "lon"), zlib=True)
    vlon = fh.createVariable("lon", "f4", ("lat", "lon"), zlib=True)
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
    vlon[:] = lon
    vlat[:] = lat
    vtime[:] = time_lp
    veofn[:] = np.arange(1, neof + 1)

    vEOF[:] = EOFs
    vPC[:]  = PCs
    vVAR[:] = var_frac  # already fraction (0-1)

    fh.close()
    print(f"Results saved to {filename}.\n")


