#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 20:38:30 2025

@author: 6008399

EOF PDV plot
"""
#%%
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
#from sklearn.linear_model import LinearRegression
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
import cartopy.mpl.ticker as cticker

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%% Read in data

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

#%%

moving_average = 0
month_start    = 1
month_end      = 12

lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory + 'EOF_PDV_SST_forward_month_1_12_moving_average_0_CESM_QE_year_600_1500_quadratic_detrend.nc')
lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory + 'EOF_PDV_SST_backward_month_1_12_moving_average_0_CESM_QE_year_2900_3800_quadratic_detrend.nc')

#%% Take first EOF for NAO

EOF_TEMP_E1 = EOF_E1[0,:,:]
EOF_TEMP_E2 = EOF_E2[0,:,:]


#%%

#Align signs using correlation of PCs (first mode)
corr = np.corrcoef(EOF_TEMP_E1, EOF_TEMP_E2)[0,1]
if corr < 0:
    print('Switching signs')
    EOF_TEMP_E2 *= -1
    PC_E2[0,:] *= -1
    
#EOF_SLP_E4 = -EOF_SLP_E4
#PC_E4 = -PC_E4

#%%    

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('Pacific SST', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.03,0.03,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) First EOF monthly SST - PI$^{on}_{QE}$ (var.ex. = '+str(int(VAR_E1[0]))+'%)')
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(0, 70)

c2 = axs[0,1].contourf(lon, lat, EOF_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.03,0.03,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) First EOF monthly SST - PI$^{off}_{QE}$ (var.ex. = '+str(int(VAR_E2[0]))+'%)')
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(0, 70)

c1 = axs[1,0].contourf(lon, lat, EOF_E1[1,:,:], transform=ccrs.PlateCarree(), levels = np.linspace(-0.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) Second EOF AMOC on (var.ex. = '+str(int(VAR_E1[1]))+'%)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(0, 70)

c2 = axs[1,1].contourf(lon, lat, EOF_E2[1,:,:], transform=ccrs.PlateCarree(), levels = np.linspace(-.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Second EOF AMOC off (var.ex. = '+str(int(VAR_E2[1]))+'%)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(0, 70)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'EOF_SST_Pacific_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show()    


#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0,0].contourf(lon, lat, EOF_TEMP_E2 - EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.01,0.01,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) Difference first EOF pattern')
axs[0,0].set_xticks(np.arange(-250,110,40), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(0, 70)

c2 = axs[0,1].contourf(lon, lat, EOF_E2[1,:,:] - EOF_E1[1,:,:], transform=ccrs.PlateCarree(), levels = np.linspace(-.01,0.01,21), cmap='RdBu_r', extend = 'both')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) Difference second EOF pattern')
axs[0,1].set_xticks(np.arange(-250,110,40), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(0, 70)

c1 = axs[1,0].contourf(lon, lat, EOF_TEMP_E2 * np.max(abs(PC_E2[0,:])) - EOF_TEMP_E1 * np.max(abs(PC_E1[0,:])), transform=ccrs.PlateCarree(), levels = np.linspace(-0.0005,0.0005,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) Difference first EOF*max(|PC|)')
axs[1,0].set_xticks(np.arange(-250,110,40), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(0, 70)

c2 = axs[1,1].contourf(lon, lat, EOF_E2[1,:,:] * np.max(abs(PC_E2[1,:])) - EOF_E1[1,:,:] * np.max(abs(PC_E1[1,:])), transform=ccrs.PlateCarree(), levels = np.linspace(-.0005,0.0005,21), cmap='RdBu_r', extend = 'both')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Difference second EOF*max(|PC|)')
axs[1,1].set_xticks(np.arange(-250,110,40), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(0, 70)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'EOF_SST_Pacific_diff_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show()


#%% plot PC's

#Central moving average
def Moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window = 20

fig, axs = plt.subplots(1, 2, figsize=(14, 4))  

plt.suptitle('First PC PDO pattern', fontsize=14)

axs[0].set_title('a) AMOC on')
axs[0].plot(time_E1 - time_E1[0], PC_E1[0,:], color='orange', alpha = 0.3, label='AMOC on')
axs[0].plot(time_E1[window//2 : -window//2 + 1] - time_E1[0], Moving_average(PC_E1[0,:], window), color='orange')
#axs[0].set_ylim(-0.2, 0.35)
axs[0].legend()

axs[1].set_title('b) AMCO offv')
axs[1].plot(time_E2 - time_E2[0], PC_E2[0,:], color='orange', alpha = 0.3, label='AMOC on')
axs[1].plot(time_E2[window//2 : -window//2 + 1] - time_E2[0], Moving_average(PC_E2[0,:], window), color='orange')
#axs[1].set_ylim(-0.2, 0.35)
axs[1].legend()

#%% MTM spectra of PDO, not standardised (i think it already is as we use the PC1 index..)

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

#Settings
fs = 12.0
NW = 2.5
Kmax = None
nsurr = 2000
ci_level = 95

# --- compute absolute PSDs + envelopes ---
f_sst_on,  S_sst_on,  ci_sst_on,  phi_sst_on  = mtm_psd_ar1_ci_abs(PC_E1,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=11)
f_sst_off, S_sst_off, ci_sst_off, phi_sst_off = mtm_psd_ar1_ci_abs(PC_E2, fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=12)

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_sst_on,  Sp_sst_on,  CIp_sst_on  = to_period_sorted(f_sst_on,  S_sst_on,  ci_sst_on[ci_level])
per_sst_off, Sp_sst_off, CIp_sst_off = to_period_sorted(f_sst_off, S_sst_off, ci_sst_off[ci_level])

#%%

period_xlim = (5, 100)
period_xlim = (5, 300)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

panels = [
    (axes[0], per_sst_on,  Sp_sst_on,  CIp_sst_on,  r"c) PI$^{on}_{QE}$ - PC1 PDO",  "c"),
    (axes[1], per_sst_off,  Sp_sst_off,  CIp_sst_off,  r"d) PI$^{off}_{QE}$ - PC1 PDO", "d")]

for ax, per, Sp, CIp, title, lab in panels:
    ax.axvspan(10, 60, alpha=0.2, color="royalblue")  #PDV band
    ax.plot(per, Sp, linewidth=1.8, color="royalblue", label="MT spectrum")
    ax.plot(per, CIp, linestyle="--", linewidth=1.2, color="red", label=f"AR(1) {ci_level}%")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.3)

axes[0].set_ylabel("Power [normalised]", fontsize =11)
#axes[1,0].set_ylabel("Power [normalised]", fontsize = 11)
axes[0].set_xlabel("Period [model years]", fontsize = 11)
axes[1].set_xlabel("Period [model years]", fontsize=11)

axes[0].legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.savefig(directory_figures + "MTM_PDO_PC1_off_on.pdf", dpi=300, bbox_inches="tight")
plt.show()

#%%

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import numpy as np

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.28, wspace=0.22)

# =========================
# Top row: EOF maps
# =========================
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=180))

for ax in [ax1, ax2]:
    ax.coastlines()
    ax.set_xticks(np.arange(-250, 110, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.set_yticks(np.arange(0, 81, 20), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.set_xlim(-80, 110)
    ax.set_ylim(5, 70)

levels_eof = np.linspace(-0.03, 0.03, 21)

c1 = ax1.contourf(
    lon, lat, EOF_TEMP_E1,
    transform=ccrs.PlateCarree(),
    levels=levels_eof, cmap='RdBu_r', extend='both')
cb1 = fig.colorbar(c1, ax=ax1, orientation='horizontal', shrink=1)
#cb1.set_label("EOF amplitude")
ax1.set_title(r'a) First EOF monthly SST - PI$^{\mathrm{on}}_{\mathrm{QE}}$ (var. ex. = ' + str(int(VAR_E1[0])) + '%)')

c2 = ax2.contourf(
    lon, lat, EOF_TEMP_E2,
    transform=ccrs.PlateCarree(),
    levels=levels_eof, cmap='RdBu_r', extend='both')
cb2 = fig.colorbar(c2, ax=ax2, orientation='horizontal', shrink=1)
#cb2.set_label("EOF amplitude")
ax2.set_title(r'b) First EOF monthly SST - PI$^{\mathrm{off}}_{\mathrm{QE}}$ (var. ex. = ' + str(int(VAR_E2[0])) + '%)')

# =========================
# Bottom row: MTM spectra
# =========================
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax3)

period_xlim = (5, 300)

panels = [
    (ax3, per_sst_on,  Sp_sst_on,  CIp_sst_on,  r"c) PI$^{\mathrm{on}}_{\mathrm{QE}}$ - PC1 PDO"),
    (ax4, per_sst_off, Sp_sst_off, CIp_sst_off, r"d) PI$^{\mathrm{off}}_{\mathrm{QE}}$ - PC1 PDO"),
]

for ax, per, Sp, CIp, title in panels:
    ax.axvspan(10, 60, alpha=0.2, color="royalblue")  #PDV band
    ax.plot(per, Sp, linewidth=1.8, color="royalblue", label="MT spectrum")
    ax.plot(per, CIp, linestyle="--", linewidth=1.2, color="red", label=f"AR(1) {ci_level}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*period_xlim)
    ax.set_ylim(5*1e-11, 1e-8)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="major", alpha=0.3)
    #ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    #ax.xaxis.set_minor_formatter(mticker.NullFormatter())

ax3.set_ylabel("Power [K$^2$ / yr$^{-1}$]", fontsize=11)
ax3.set_xlabel("Period [model years]", fontsize=11)
ax4.set_xlabel("Period [model years]", fontsize=11)
ax3.legend(frameon=False, loc="lower right")

plt.savefig(directory_figures + "EOF_PDO_MTM_combined.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
