#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 19:48:44 2026

@author: 6008399

Surface heat flux

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
from scipy.interpolate import interp1d
from amoc_tools import *

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

def ReadinData(filename):

	fh = netcdf.Dataset(filename, 'r')

	lat	       = fh.variables['lat'][:,0:100]     		  #Latitudes (degrees N)
	lon	       = fh.variables['lon'][:,0:100]     	      #Longitudes(degrees E)
	time	   = fh.variables['time'][:]		              #Time (model year)
	time_month = fh.variables['time_month'][:]		          #Time (model year)
	SHF        = fh.variables['SHF'][:,:,0:100]     	  #surface heat flux (W/m2) -> positive into the ocean
	SHF_month  = fh.variables['SHF_month'][:,:,0:100]    #Monthly surface heat flux (W/m2)

	fh.close()

	return lat, lon, time, time_month, SHF, SHF_month

def ReadinDataSST(filename):

	fh = netcdf.Dataset(filename, 'r')

	lat	       = fh.variables['lat'][:, 80:80+100]     		  #Latitudes (degrees N)
	lon	       = fh.variables['lon'][:, 80:80+100]     	      #Longitudes(degrees E)
	area	   = fh.variables['area'][:, 80:80+100] 
	time	   = fh.variables['time'][:]		              #Time (model year)
	time_month = fh.variables['time_month'][:]		          #Time (model year)
	SST        = fh.variables['SST'][:, :, 80:80+100]     	  #surface heat flux (W/m2)
	SST_month  = fh.variables['SST_month'][:,:,80:80+100]    #Monthly surface heat flux (W/m2)

	fh.close()

	return lat, lon, area, time, time_month, SST, SST_month

def ReadinDataIFRAC(filename):

	fh = netcdf.Dataset(filename, 'r')

	lat	       = fh.variables['lat'][:,0:100]     		  #Latitudes (degrees N)
	lon	       = fh.variables['lon'][:,0:100]     	      #Longitudes(degrees E)
	time	   = fh.variables['time'][:]	
	area	   = fh.variables['area'][:,0:100] 	              #Time (model year)
	time_month = fh.variables['time_month'][:]		          #Time (model year)
	IFRAC        = fh.variables['ice'][:,:,0:100]     	  #surface heat flux (W/m2)
	IFRAC_month  = fh.variables['ice_month'][:,:,0:100]    #Monthly surface heat flux (W/m2)

	fh.close()

	return lat, lon, area, time, time_month, IFRAC, IFRAC_month

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

#%%

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------	

lat, lon, time_forward, time_month_forward, SHF_forward, SHF_month_forward 	= ReadinData(directory_data+'/SHF_Atlantic_year_600-1500_month_1-12_QE.nc')
lat, lon, time_backward, time_month_backward, SHF_backward, SHF_month_backward 	= ReadinData(directory_data+'/SHF_Atlantic_year_2900-3800_month_1-12_QE.nc')

lat_SST, lon_SST, area_SST, time_forward, time_month_forward, SST_forward, SST_month_forward 	= ReadinDataSST(directory_data+'/SST_Atlantic_year_600-1500_month_1-12_QE.nc')
lat_SST, lon_SST, area_SST, time_backward, time_month_backward, SST_backward, SST_month_backward 	= ReadinDataSST(directory_data+'/SST_Atlantic_year_2900-3800_month_1-12_QE.nc')

lat, lon, area, time_forward, time_month_forward, IFRAC_forward, IFRAC_month_forward 	= ReadinDataIFRAC(directory_data+'/IFRAC_Atlantic_year_600-1500_month_1-12_QE.nc')
lat, lon, area, time_backward, time_month_backward, IFRAC_backward, IFRAC_month_backward 	= ReadinDataIFRAC(directory_data+'/IFRAC_Atlantic_year_2900-3800_month_1-12_QE.nc')

#%%

plt.figure()
plt.pcolormesh(~area.mask)
plt.title("Unmasked area cells (should be NA only)")
plt.show()

plt.figure()
plt.contourf(lon, lat, area)#, levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Unmasked area cells (should be NA only)")
plt.show()

plt.figure()
plt.contourf(lon_SST, lat_SST, SST_month_forward[0], levels=np.linspace(-200, 200, 21), cmap='Spectral_r', extend='both')

#%% Use NA mask to mask out non-NA grid cells in SHF (and later SST and IFRAC)

lat_min = 0
lat_max = 60

mask2d = north_atlantic_mask(lat, lon, lat_min=lat_min, lat_max=lat_max)
area = ma.masked_where(~mask2d, area)

mask3d = np.broadcast_to(~mask2d, SHF_month_forward.shape)
SHF_month_forward= ma.masked_where(mask3d, SHF_month_forward)
SHF_month_backward = ma.masked_where(mask3d, SHF_month_backward)

IFRAC_month_forward = ma.masked_where(mask3d, IFRAC_month_forward)
IFRAC_month_backward = ma.masked_where(mask3d, IFRAC_month_backward)

SST_month_forward = ma.masked_where(mask3d, SST_month_forward)
SST_month_backward = ma.masked_where(mask3d, SST_month_backward)

# mask3d = np.broadcast_to(~mask2d, SHF_forward.shape)
# SHF_month_forward= ma.masked_where(mask3d, SHF_forward)
# SHF_month_backward = ma.masked_where(mask3d, SHF_backward)

# IFRAC_month_forward = ma.masked_where(mask3d, IFRAC_forward)
# IFRAC_month_backward = ma.masked_where(mask3d, IFRAC_backward)

# SST_month_forward = ma.masked_where(mask3d, SST_forward)
# SST_month_backward = ma.masked_where(mask3d, SST_backward)


#%%

plt.figure()
plt.contourf(lon, lat, SHF_month_forward[0], levels=np.linspace(-200, 200, 21), cmap='Spectral_r', extend='both')

plt.figure()
plt.contourf(lon_SST, lat_SST, SST_month_forward[0], levels=np.linspace(-200, 200, 21), cmap='Spectral_r', extend='both')

plt.figure()
plt.contourf(lon, lat, area)

#%% De-seasonlise (monthly) and detrend 

SHF_DS_DT_forward = ma.masked_all(SHF_month_forward.shape)
SHF_DS_DT_backward = ma.masked_all(SHF_month_backward.shape)

SST_DS_DT_forward = ma.masked_all(SST_month_forward.shape)
SST_DS_DT_backward = ma.masked_all(SST_month_backward.shape)

for lat_i in range(len(lat)):
    for lon_i in range(len(lon[0])):

            if ma.is_masked(SHF_month_forward[0, lat_i, lon_i]):
                continue

            SHF_DS_forward = MonthRemover(time_month_forward, SHF_month_forward[:, lat_i, lon_i])
            SHF_DS_backward = MonthRemover(time_month_backward, SHF_month_backward[:, lat_i, lon_i])

            SHF_DS_DT_forward[:, lat_i, lon_i] = TrendRemover(time_month_forward, SHF_DS_forward, 2)
            SHF_DS_DT_backward[:, lat_i, lon_i] = TrendRemover(time_month_backward, SHF_DS_backward, 2)

            #SHF_DS_DT_forward[:, lat_i, lon_i] = TrendRemover(time_forward, SHF_month_forward[:, lat_i, lon_i], 2)
            #SHF_DS_DT_backward[:, lat_i, lon_i] = TrendRemover(time_backward, SHF_month_backward[:,lat_i, lon_i], 2)

            if ma.is_masked(SST_month_forward[0, lat_i, lon_i]):
                continue

            SST_DS_forward = MonthRemover(time_month_forward, SST_month_forward[:, lat_i, lon_i])
            SST_DS_backward = MonthRemover(time_month_backward, SST_month_backward[:, lat_i, lon_i])

            SST_DS_DT_forward[:, lat_i, lon_i] = TrendRemover(time_month_forward, SST_DS_forward, 2)
            SST_DS_DT_backward[:, lat_i, lon_i] = TrendRemover(time_month_backward, SST_DS_backward, 2)

            #SST_DS_DT_forward[:, lat_i, lon_i] = TrendRemover(time_forward, SST_month_forward[:, lat_i, lon_i], 2)
            #SST_DS_DT_backward[:, lat_i, lon_i] = TrendRemover(time_backward, SST_month_backward[:, lat_i, lon_i], 2)

#%% Area-weighted NA mean (area has same mask as other arrays)

SHF_NA_forward = ma.sum(SHF_DS_DT_forward * area, axis=(1,2)) / ma.sum(area)
SHF_NA_backward = ma.sum(SHF_DS_DT_backward * area, axis=(1,2)) / ma.sum(area)

SST_NA_forward = ma.sum(SST_DS_DT_forward * area, axis=(1,2)) / ma.sum(area)
SST_NA_backward = ma.sum(SST_DS_DT_backward * area, axis=(1,2)) / ma.sum(area)

IFRAC_NA_forward  = ma.sum(IFRAC_month_forward  * area, axis=(1,2)) / ma.sum(area)
IFRAC_NA_backward = ma.sum(IFRAC_month_backward * area, axis=(1,2)) / ma.sum(area)

#%%

plt.figure()
plt.contourf(lon, lat, np.mean(IFRAC_month_forward, axis=0))

plt.figure()
plt.contour(lon, lat, np.mean(IFRAC_month_backward, axis=0), levels=np.linspace(0.15, 0.16, 1))
plt.colorbar()

#%%

fig, ax1 = plt.subplots()

ax1.plot(SHF_NA_forward, label="SHF", color="tab:blue")
ax1.set_ylabel("SHF (W/m²)", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(SST_NA_forward, label="SST", color="tab:orange")
ax2.plot(IFRAC_NA_forward, label="IFRAC", color="tab:green")
ax2.set_ylabel("SST (°C) / IFRAC", color="tab:orange")

ax1.set_title("PI$_{QE}^{on}$")
fig.legend(loc="upper left")
plt.show()

fig, ax1 = plt.subplots()

ax1.plot(SHF_NA_backward, label="SHF", color="tab:blue")
ax1.set_ylabel("SHF (W/m²)", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(SST_NA_backward, label="SST", color="tab:orange")
ax2.plot(IFRAC_NA_backward, label="IFRAC", color="tab:green")
ax2.set_ylabel("SST (°C) / IFRAC", color="tab:orange")

ax1.set_title("PI$_{QE}^{off}$")
fig.legend(loc="upper left")
plt.show()

#%%

import numpy as np

def running_mean(x, window):
    """
    Simple running mean (centered), ignores NaNs.
    """
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)

    half = window // 2

    for i in range(len(x)):
        i0 = max(0, i - half)
        i1 = min(len(x), i + half + 1)
        out[i] = np.nanmean(x[i0:i1])

    return out

# High-pass (noise)
SHF_HP_forward  = SHF_NA_forward  - running_mean(SHF_NA_forward, 60)
SHF_HP_backward = SHF_NA_backward - running_mean(SHF_NA_backward, 60)

# Low-pass
SHF_LP_forward  = running_mean(SHF_NA_forward, 120)
SHF_LP_backward = running_mean(SHF_NA_backward, 120)

# Variances
var_HP_f = np.nanvar(SHF_HP_forward)
var_HP_b = np.nanvar(SHF_HP_backward)

var_LP_f = np.nanvar(SHF_LP_forward)
var_LP_b = np.nanvar(SHF_LP_backward)

#%%

print(var_HP_f, var_LP_f)
print(var_HP_b, var_LP_b)

#%% Open ocean only statistics

open_mask_f = (IFRAC_month_forward < 0.15)  # shape (time,lat,lon). 0 is ice, 1 is no ice. 
open_mask_b = (IFRAC_month_backward < 0.15)

x = np.array([0.1, 0.5, 0.0])
mask = (x < 0.15)
print(mask)

plt.figure()
plt.contourf(lon, lat, IFRAC_month_forward[0], levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Fraction of time each grid cell is open ocean (IFRAC < 0.15) - PI_QE_on")
plt.colorbar(label="Fraction of time open")
plt.show()

plt.figure()
plt.contourf(lon, lat, open_mask_f[0], levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Fraction of time each grid cell is open ocean (IFRAC < 0.15) - PI_QE_on")
plt.colorbar(label="Fraction of time open")
plt.show()

plt.figure()
plt.contourf(lon, lat, open_mask_b[0], levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Fraction of time each grid cell is open ocean (IFRAC < 0.15) - PI_QE_on")
plt.colorbar(label="Fraction of time open")
plt.show()

#%%

nt = IFRAC_month_forward.shape[0]

area3 = ma.array(
    np.broadcast_to(area.data, (nt,) + area.shape).copy(),
    mask=np.broadcast_to(ma.getmaskarray(area), (nt,) + area.shape).copy())

area_open_f = ma.array(area3, mask=area3.mask | (~open_mask_f))
area_open_b = ma.array(area3, mask=area3.mask | (~open_mask_b))

#%%

plt.figure()
plt.contourf(lon, lat, area_open_f[0])#, levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Area of open ocean grid cells (IFRAC < 0.15) - PI_QE_on")
plt.colorbar(label="Area (m²)")
plt.show()

plt.figure()
plt.contourf(lon, lat, area3[1])#, levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Area3[0]")
plt.colorbar(label="Area (m²)")
plt.show()

plt.figure()
plt.contourf(lon, lat, area)#, levels=np.linspace(0, 1, 11), cmap='viridis', extend='both')
plt.title("Area")
plt.colorbar(label="Area (m²)")
plt.show()

#%%

SHF_NA_open_f = ma.sum(SHF_DS_DT_forward * area_open_f, axis=(1,2)) / ma.sum(area_open_f, axis=(1,2))
SHF_NA_open_b = ma.sum(SHF_DS_DT_backward * area_open_b, axis=(1,2)) / ma.sum(area_open_b, axis=(1,2))

SST_NA_open_f = ma.sum(SST_DS_DT_forward * area_open_f, axis=(1,2)) / ma.sum(area_open_f, axis=(1,2))
SST_NA_open_b = ma.sum(SST_DS_DT_backward * area_open_b, axis=(1,2)) / ma.sum(area_open_b, axis=(1,2))

#%%

# High-pass (noise)
SHF_HP_open_f  = SHF_NA_open_f  - running_mean(SHF_NA_open_f, 60)
SHF_HP_open_b = SHF_NA_open_b - running_mean(SHF_NA_open_b, 60)

# Low-pass
SHF_LP_open_f  = running_mean(SHF_NA_open_f, 120)
SHF_LP_open_b = running_mean(SHF_NA_open_b, 120)

#%% Variances
var_HP_open_f = np.nanvar(SHF_HP_open_f)
var_HP_open_b = np.nanvar(SHF_HP_open_b)

var_LP_open_f = np.nanvar(SHF_LP_open_f)
var_LP_open_b = np.nanvar(SHF_LP_open_b)

print(var_HP_open_f, var_LP_open_f)
print(var_HP_open_b, var_LP_open_b)

#%% Open ocean fraction

# Boolean open mask
open_mask_f = (IFRAC_month_forward < 0.3)
open_mask_b = (IFRAC_month_backward < 0.3)

# Total NA area (constant)
NA_area = ma.sum(area)

# Open-ocean area at each time step
open_area_f = ma.sum(area * open_mask_f, axis=(1,2))
open_area_b = ma.sum(area * open_mask_b, axis=(1,2))

# Fraction
frac_open_f = open_area_f / NA_area
frac_open_b = open_area_b / NA_area

#%%
plt.figure()
plt.plot(frac_open_f, label="PI_QE_on")
#plt.plot(frac_open_b, label="PI_QE_off")
plt.ylabel("Open-ocean fraction of NA")
plt.xlabel("Time (months)")
plt.title("NA Open-Ocean Area Fraction (IFRAC < 0.15)")
plt.legend()
plt.show()

#%% damping coefficient

coef_open_f = np.polyfit(SST_NA_open_f, SHF_NA_open_f, 1)
coef_open_b = np.polyfit(SST_NA_open_b, SHF_NA_open_b, 1)

lambda_open_f = -coef_open_f[0]
lambda_open_b = -coef_open_b[0]

print('Lambda (open ocean) - AMOC forward:', lambda_open_f)
print('Lambda (open ocean) - AMOC backward:', lambda_open_b)

coef_f = np.polyfit(SST_NA_forward, SHF_NA_forward, 1)
coef_b = np.polyfit(SST_NA_backward, SHF_NA_backward, 1)

lambda_f = -coef_f[0]
lambda_b = -coef_b[0]

print('Lambda (NA) - AMOC forward:', lambda_f)
print('Lambda (NA) - AMOC backward:', lambda_b)

lag = 3  # months

coef_lagged_f = np.polyfit(SST_NA_forward[:-lag], SHF_NA_forward[lag:], 1)
coef_lagged_b = np.polyfit(SST_NA_backward[:-lag], SHF_NA_backward[lag:], 1)

lambda_lagged_f = -coef_lagged_f[0]
lambda_lagged_b = -coef_lagged_b[0]

print(f'Lagged Lambda (open ocean, lag={lag} month) - AMOC forward:', lambda_lagged_f)
print(f'Lagged Lambda (open ocean, lag={lag} month) - AMOC backward:', lambda_lagged_b)

#%% Spatial mean

lag = 1  # months; use lag=0 for simultaneous regression

lambda_forward  = ma.masked_all(SHF_month_forward.shape[1:])
lambda_backward = ma.masked_all(SHF_month_backward.shape[1:])

for lat_i in range(len(lat)):
    for lon_i in range(len(lon[0])):

        if ma.is_masked(SHF_month_forward[0, lat_i, lon_i]):
            continue
        if ma.is_masked(SST_month_forward[0, lat_i, lon_i]):
            continue

        # Forward
        sst_f = SST_DS_DT_forward[:, lat_i, lon_i]
        shf_f = SHF_DS_DT_forward[:, lat_i, lon_i]

        # Backward
        sst_b = SST_DS_DT_backward[:, lat_i, lon_i]
        shf_b = SHF_DS_DT_backward[:, lat_i, lon_i]

        if lag == 0:
            x_f, y_f = sst_f, shf_f
            x_b, y_b = sst_b, shf_b
        else:
            x_f, y_f = sst_f[:-lag], shf_f[lag:]
            x_b, y_b = sst_b[:-lag], shf_b[lag:]

        # Remove masked / invalid values
        mask_f = np.isfinite(x_f) & np.isfinite(y_f)
        mask_b = np.isfinite(x_b) & np.isfinite(y_b)

        if np.sum(mask_f) > 2:
            coef_forward = np.polyfit(x_f[mask_f], y_f[mask_f], 1)
            lambda_forward[lat_i, lon_i] = -coef_forward[0]

        if np.sum(mask_b) > 2:
            coef_backward = np.polyfit(x_b[mask_b], y_b[mask_b], 1)
            lambda_backward[lat_i, lon_i] = -coef_backward[0]

#%%

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

fig, axs = plt.subplots(1, 3, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:
    ax.coastlines()
    ax.set_xticks(np.arange(-90, 31, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 70, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())   

cs1 = axs[0].contourf(lon, lat, lambda_forward, levels=np.linspace(0, 20, 21), cmap='Spectral_r', extend='both')
fig.colorbar(cs1, ax=axs[0], orientation='horizontal', shrink=0.8, label='λ (W m⁻² K⁻¹)')
axs[0].contour(lon, lat, np.mean(IFRAC_month_forward, axis=0), levels=np.linspace(0.15, 0.16, 1), colors='black', linewidths=2)
axs[0].set_title('a) Damping coefficient λ - lag '+str(lag)+' months (PI$^{\mathrm{on}}_{\mathrm{QE}}$)')
axs[0].set_xlim(-70, 0)
axs[0].set_ylim(0, 60)

cs2 = axs[1].contourf(lon, lat, lambda_backward, levels=np.linspace(0, 20, 21), cmap='Spectral_r', extend='both')
fig.colorbar(cs2, ax=axs[1], orientation='horizontal', shrink=0.8, label='λ (W m⁻² K⁻¹)')
axs[1].contour(lon, lat, np.mean(IFRAC_month_backward, axis=0), levels=np.linspace(0.15, 0.16, 1), colors='black', linewidths=2)
axs[1].set_title('b) Damping coefficient λ - lag '+str(lag)+' months (PI$^{\mathrm{off}}_{\mathrm{QE}}$)')
axs[1].set_xlim(-70, 0)
axs[1].set_ylim(0, 60)

lambda_diff = lambda_backward - lambda_forward
cs3 = axs[2].contourf(lon, lat, lambda_diff, levels=np.linspace(-10, 10, 21), cmap='RdBu_r', extend='both')
axs[2].contour(lon, lat, np.mean(IFRAC_month_backward, axis=0), levels=np.linspace(0.15, 0.16, 1), colors='black', linewidths=2)
fig.colorbar(cs3, ax=axs[2], orientation='horizontal', shrink=0.8, label='λ (W m⁻² K⁻¹)')
axs[2].set_title('c) Difference in λ (PI$^{\mathrm{off}}_{\mathrm{QE}}$ - PI$^{\mathrm{on}}_{\mathrm{QE}}$)')
axs[2].set_xlim(-70, 0)
axs[2].set_ylim(0, 60)

plt.tight_layout()
plt.savefig(directory_figures + 'SHF_damping_coefficient_lambda_lag_'+str(lag)+'months.pdf', dpi=300)
plt.show()

#%% Basin means 

print('Damping coefficient λ (NA) - AMOC forward - lag ' + str(lag) + ' months:', np.sum(lambda_forward * area) / np.sum(area))
print('Damping coefficient λ (NA) - AMOC backward - lag ' + str(lag) + ' months:', np.sum(lambda_backward * area) / np.sum(area))

#only open ocean grid cells
print('Damping coefficient λ (NA open ocean) - AMOC forward - lag ' + str(lag) + ' months:', np.sum(lambda_forward * area_open_f) / np.sum(area_open_f))
print('Damping coefficient λ (NA open ocean) - AMOC backward - lag ' + str(lag) + ' months:', np.sum(lambda_backward * area_open_b) / np.sum(area_open_b))
print('Reduction in lambda AMOC backward compared to AMOC forward (open ocean):', (np.sum(lambda_forward * area_open_f) / np.sum(area_open_f) - np.sum(lambda_backward * area_open_b) / np.sum(area_open_b)) / (np.sum(lambda_forward * area_open_f) / np.sum(area_open_f)) * 100, '%')

print('Damping coefficient λ (NA-basin mean first) - AMOC forward - lag ' + str(lag) + ' months:', lambda_open_f)
print('Damping coefficient λ (NA-basin mean first) - AMOC backward - lag ' + str(lag) + ' months:', lambda_open_b)

#%%
#Restricting to north of 20N
lat_mask = (lat > 20)
area_north = ma.masked_where(~lat_mask, area)
lambda_north_f = np.sum(lambda_forward * area_north) / np.sum(area_north)
lambda_north_b = np.sum(lambda_backward * area_north) / np.sum(area_north)

plt.figure()
plt.contourf(lon, lat, area_north)

print('Damping coefficient λ (NA north of 20N) - AMOC forward - lag ' + str(lag) + ' months:', lambda_north_f)
print('Damping coefficient λ (NA north of 20N) - AMOC backward - lag ' + str(lag) + ' months:', lambda_north_b)
print('Reduction in lambda AMOC backward compared to AMOC forward (NA north of 20N):', (lambda_north_f - lambda_north_b) / lambda_north_f * 100, '%')

#%%

print('Correlation (NA) - AMOC forward:', np.corrcoef(SST_NA_forward, SHF_NA_forward)[0, 1])
print('Correlation (NA) - AMOC backward:', np.corrcoef(SST_NA_backward, SHF_NA_backward)[0, 1])

print('Correlation (NA) - AMOC forward lag:', np.corrcoef(SST_NA_forward[:-lag], SHF_NA_forward[lag:])[0, 1])
print('Correlation (NA) - AMOC backward lag:', np.corrcoef(SST_NA_backward[:-lag], SHF_NA_backward[lag:])[0, 1])

#%%

plt.figure()
plt.scatter(SST_NA_forward[:-lag], SHF_NA_forward[lag:], alpha=0.3)

#%%

from scipy import stats

r, p = stats.pearsonr(SST_NA_forward, SHF_NA_forward)
print('P-value (NA) - AMOC forward:', p)

r, p = stats.pearsonr(SST_NA_backward, SHF_NA_backward)
print('P-value (NA) - AMOC backward:', p)

#%% Mean and std plots 

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

fig, axs = plt.subplots(
    2, 3, figsize=(14, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

for ax in axs.flat:
    ax.coastlines()
    ax.set_xticks(np.arange(-90, 31, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(20, 70, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

# Mean MLD
c1 = axs[0,0].contourf(
    lon, lat, np.nanmean(SHF_forward, axis=0),levels=np.linspace(-200, 200, 21),cmap='Spectral_r', extend='both',
    transform=ccrs.PlateCarree())
axs[0,0].set_title('a) Mean SHF (PI$^{on}_{QE}$)')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal',shrink=0.8,
             label='m')

c2 = axs[0,1].contourf(
    lon, lat, np.nanmean(SHF_backward, axis=0),levels=np.linspace(-200, 200, 21),cmap='Spectral_r', extend='both',
    transform=ccrs.PlateCarree())
axs[0,1].set_title('b) Mean SHF (PI$^{off}_{QE}$)')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal',shrink=0.8,
             label='m')

c3 = axs[0,2].contourf(
    lon, lat, np.nanmean(SHF_backward, axis=0) - np.nanmean(SHF_forward, axis=0),levels=np.linspace(-300, 300, 21),cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree())
axs[0,2].set_title('c) Difference mean SHF (PI$^{off}_{QE}$ - PI$^{on}_{QE}$)')
fig.colorbar(c3, ax=axs[0,2], orientation='horizontal',shrink=0.8,
             label='m')

# STD MLD
c1 = axs[1,0].contourf(
    lon, lat, np.nanstd(SHF_DS_DT_forward, axis=0),levels=np.linspace(-100, 100, 21),cmap='Spectral_r', extend='both',
    transform=ccrs.PlateCarree())
axs[1,0].set_title('a) Std SHF (PI$^{on}_{QE}$)')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal',shrink=0.8,
             label='m')

c2 = axs[1,1].contourf(
    lon, lat, np.nanstd(SHF_DS_DT_backward, axis=0),levels=np.linspace(-100, 100, 21),cmap='Spectral_r', extend='both',
    transform=ccrs.PlateCarree())
axs[1,1].set_title('b) Std SHF (PI$^{off}_{QE}$)')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal',shrink=0.8,
             label='m')

c3 = axs[1,2].contourf(
    lon, lat, np.nanstd(SHF_DS_DT_backward, axis=0) - np.nanstd(SHF_DS_DT_forward, axis=0),levels=np.linspace(-50, 50, 21),cmap='RdBu_r', extend='both',
    transform=ccrs.PlateCarree())
axs[1,2].set_title('c) Difference std SHF (PI$^{off}_{QE}$ - PI$^{on}_{QE}$)')
fig.colorbar(c3, ax=axs[1,2], orientation='horizontal',shrink=0.8,
             label='m')

#plt.suptitle('Regression of SST and Subsurface Temperature onto AMOC Strength', fontsize=14)

plt.tight_layout()
plt.savefig(directory_figures + 'Mean_std_SHF_amoc_on_off.pdf')
plt.show()

#%%

import numpy as np

def reg_stats(x, y):
    """Return slope, intercept, r, N for y = slope*x + intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    N = x.size
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r, N

x = np.asarray(x.filled(np.nan))

import matplotlib.pyplot as plt

def regression_panel(ax, x, y, title, subsample=5):
    # Convert masked arrays to NaNs if needed
    if hasattr(x, "filled"): x = x.filled(np.nan)
    if hasattr(y, "filled"): y = y.filled(np.nan)

    # subsample for plotting clarity
    xs = x[::subsample]
    ys = y[::subsample]

    slope, intercept, r, N = reg_stats(x, y)
    lam = -slope  # your definition

    # scatter
    ax.scatter(xs, ys, s=6, alpha=0.25)

    # regression line over x-range
    xmin, xmax = np.nanpercentile(x, [1, 99])
    xx = np.linspace(xmin, xmax, 200)
    yy = slope*xx + intercept
    ax.plot(xx, yy, linewidth=2)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("NA SST anomaly (°C)")
    ax.set_ylabel("NA SHF anomaly (W m$^{-2}$)")

    ax.text(
        0.02, 0.98,
        f"slope dQ/dT = {slope:.2f} W m$^{{-2}}$ K$^{{-1}}$\n"
        f"λ = -slope = {lam:.2f} W m$^{{-2}}$ K$^{{-1}}$\n"
        f"r = {r:.2f},  N = {N}",
        transform=ax.transAxes, va="top", ha="left"
    )

    return slope, intercept, r, N

# --- Choose which series: full NA or open-ocean NA ---
# Full NA:
x_on,  y_on  = SST_NA_forward,  SHF_NA_forward
x_off, y_off = SST_NA_backward, SHF_NA_backward

# OR Open-ocean NA:
# x_on,  y_on  = SST_NA_open_f,  SHF_NA_open_f
# x_off, y_off = SST_NA_open_b,  SHF_NA_open_b

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

regression_panel(axes[0], x_on,  y_on,  r"AMOC on")
regression_panel(axes[1], x_off, y_off, r"AMOC off")

fig.suptitle("Surface heat flux damping: regression of SHF anomalies on SST anomalies", y=1.02)
fig.tight_layout()
plt.show()

#%%

def regression_panel_hexbin(ax, x, y, title, gridsize=60):
    if hasattr(x, "filled"): x = x.filled(np.nan)
    if hasattr(y, "filled"): y = y.filled(np.nan)

    slope, intercept, r, N = reg_stats(x, y)
    lam = -slope

    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1)  # no explicit colors
    xmin, xmax = np.nanpercentile(x, [1, 99])
    xx = np.linspace(xmin, xmax, 200)
    ax.plot(xx, slope*xx + intercept, linewidth=2)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("NA SST anomaly (°C)")
    ax.set_ylabel("NA SHF anomaly (W m$^{-2}$)")

    ax.text(
        0.02, 0.98,
        f"dQ/dT = {slope:.2f}\nλ = {lam:.2f}\nr = {r:.2f}, N={N}",
        transform=ax.transAxes, va="top", ha="left"
    )

    return hb

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
hb0 = regression_panel_hexbin(axes[0], x_on,  y_on,  "AMOC on")
hb1 = regression_panel_hexbin(axes[1], x_off, y_off, "AMOC off")
fig.colorbar(hb1, ax=axes, label="Counts")  # one shared colorbar
fig.tight_layout()
plt.show()


# %%
