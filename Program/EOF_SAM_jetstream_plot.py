#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:52:15 2026

@author: 6008399

Plot of EOF SAM with westterlies 

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#Making pathway to folder with all data
#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%%

def PeriodicBoundaries3D(lon, lat, field, lon_grids = 1):
    """Add periodic zonal boundaries for 3D field"""

    #Empty field with additional zonal boundaries
    lon_2                   = np.zeros(len(lon) + lon_grids * 2)
    field_2                 = ma.masked_all((len(field), len(lat), len(lon_2)))

    #Get the left boundary, which is the right boundary of the original field
    lon_2[:lon_grids]      	 = lon[-lon_grids:] - 360.0
    field_2[:, :, :lon_grids]= field[:, :, -lon_grids:]

    #Same for the right boundary
    lon_2[-lon_grids:]        = lon[:lon_grids] + 360.0
    field_2[:, :, -lon_grids:]= field[:, :, :lon_grids]

    #And the complete field
    lon_2[lon_grids:-lon_grids]             = lon
    field_2[:, :, lon_grids:-lon_grids]     = field

    return lon_2, field_2


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

	lon, EOF    = PeriodicBoundaries3D(lon, lat, EOF)

	return lon, lat, eof, time, PC, VAR, EOF

#%%

season = "DJF"   # choose: "JJA" or "DJF"

if season == 'JJA':
    print('Season: JJA')
    lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory_data + 'EOF_SAM_SLP_E1_month_6_8_detrend1_CESM_year_999_1100.nc')
    lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory_data + 'EOF_SAM_SLP_E2_month_6_8_detrend1_CESM_year_1899_1999.nc')
    lon, lat, eof_E3, time_E3, PC_E3, VAR_E3, EOF_E3		= ReadinData(directory_data + 'EOF_SAM_SLP_E3_month_6_8_detrend1_CESM_year_3299_3400.nc')
    lon, lat, eof_E4, time_E4, PC_E4, VAR_E4, EOF_E4		= ReadinData(directory_data + 'EOF_SAM_SLP_E4_month_6_8_detrend1_CESM_year_4199_4299.nc')

if season == 'DJF':
    print('Season: DJF')
    lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory_data + 'EOF_SAM_SLP_E1_month_12_14_detrend1_CESM_year_999_1099.nc')
    lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory_data + 'EOF_SAM_SLP_E2_month_12_14_detrend1_CESM_year_1899_1999.nc')
    lon, lat, eof_E3, time_E3, PC_E3, VAR_E3, EOF_E3		= ReadinData(directory_data + 'EOF_SAM_SLP_E3_month_12_14_detrend1_CESM_year_3299_3400.nc')
    lon, lat, eof_E4, time_E4, PC_E4, VAR_E4, EOF_E4		= ReadinData(directory_data + 'EOF_SAM_SLP_E4_month_12_14_detrend1_CESM_year_4199_4299.nc')

#%% Take first EOF for SAM

EOF_SLP_E1 = EOF_E1[0,:,:]
EOF_SLP_E2 = EOF_E2[0,:,:]
EOF_SLP_E3 = EOF_E3[0,:,:]
EOF_SLP_E4 = EOF_E4[0,:,:]

#%%

#Align signs using correlation of PCs (first mode)
corr = np.corrcoef(EOF_SLP_E1, EOF_SLP_E4)[0,1]
if corr < 0:
    print('Switching signs')
    EOF_SLP_E4 *= -1
    PC_E4[0,:] *= -1
    
corr = np.corrcoef(EOF_SLP_E2, EOF_SLP_E3)[0,1]
if corr < 0:
    print('Switching signs')
    EOF_SLP_E3 *= -1
    PC_E3[0,:] *= -1
    
EOF_SLP_E4 = -EOF_SLP_E4
PC_E4 = -PC_E4

#%%    

fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

for ax in axs.flat:  
    ax.coastlines()
    ax.set_extent([-185, 190, -90, -30], crs=ccrs.PlateCarree())  

    gl = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)
    gl.top_labels = False  
    gl.right_labels = False  
    gl.xlabel_style = {'size': 10, 'color': 'black'}  
    gl.ylabel_style = {'size': 10, 'color': 'black'}  

data = [
    (EOF_SLP_E1 * np.std((PC_E1[0,:])), 'a) EOF1 JJA SLP - PI$^{on}_{18}$ (var.ex. = '+str(int(VAR_E1[0]*100))+'%)', np.linspace(-4, 4, 21)),
    (EOF_SLP_E4 * np.std((PC_E4[0,:])), 'b) EOF1 JJA SLP - PI$^{off}_{18}$ (var.ex. = '+str(int(VAR_E4[0]*100))+'%)', np.linspace(-4, 4, 21)),
    (EOF_SLP_E2 * np.std((PC_E2[0,:])), 'c) EOF1 JJA SLP - PI$^{on}_{45}$ (var.ex. = '+str(int(VAR_E2[0]*100))+'%)', np.linspace(-4, 4, 21)),
    (EOF_SLP_E3 * np.std((PC_E3[0,:])), 'd) EOF1 JJA SLP - PI$^{off}_{45}$ (var.ex. = '+str(int(VAR_E3[0]*100))+'%)', np.linspace(-4, 4, 21))
]

for ax, (EOF, title, levels) in zip(axs.flat, data):
    contour = ax.contourf(
        lon, lat, EOF, transform=ccrs.PlateCarree(),
        levels=levels, cmap='RdBu_r', extend='both'
    )
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.8)
    #cbar.set_label('EOF Value', fontsize=10)
    ax.set_title(title, fontsize=12)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_SAM_EOF_SM_monthly.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

for ax in axs.flat:  
    ax.coastlines()
    ax.set_extent([-185, 190, -90, -30], crs=ccrs.PlateCarree())  

    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)
    gl.top_labels = False  
    gl.right_labels = False  
    gl.xlabel_style = {'size': 10, 'color': 'black'}  
    gl.ylabel_style = {'size': 10, 'color': 'black'}  

data = [
    (EOF_E1[1,:,:] * np.std((PC_E1[1,:])), 'a) EOF2 monthly SLP - PI$^{on}_{18}$ (var.ex. = '+str(int(VAR_E1[1]*100))+'%)', np.linspace(-7, 7, 21)),
    (EOF_E4[1,:,:] * np.std((PC_E4[1,:])), 'b) EOF2 monthly SLP - PI$^{off}_{18}$ (var.ex. = '+str(int(VAR_E4[1]*100))+'%)', np.linspace(-7, 7, 21)),
    (-EOF_E2[1,:,:] * np.std((PC_E2[1,:])), 'c) EOF2 monthly SLP - PI$^{on}_{45}$ (var.ex. = '+str(int(VAR_E2[1]*100))+'%)', np.linspace(-7, 7, 21)),
    (EOF_E3[1,:,:] * np.std((PC_E3[1,:])), 'd) EOF2 monthly SLP - PI$^{off}_{45}$ (var.ex. = '+str(int(VAR_E3[1]*100))+'%)', np.linspace(-7, 7, 21))
]

for ax, (EOF, title, levels) in zip(axs.flat, data):
    contour = ax.contourf(
        lon, lat, EOF, transform=ccrs.PlateCarree(),
        levels=levels, cmap='RdBu_r', extend='both'
    )
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.8)
    #cbar.set_label('EOF Value', fontsize=10)
    ax.set_title(title, fontsize=12)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_SAM_EOF_SM.pdf')
plt.show() 

#%%

eof1_zonal = np.mean(EOF_SLP_E1, axis=1)

plt.figure()
plt.plot(eof1_zonal, lat)
plt.axvline(0, linestyle='--')
plt.xlabel("EOF1 SLP anomaly")
plt.ylabel("Latitude")
plt.title("Zonal mean EOF1 (SAM structure)")
plt.gca().invert_yaxis()
plt.show()

#%%

def get_season(data, season="JJA"):
    """
    Convert monthly data (time, lat, lon) into seasonal data
    with shape (nseason, 3, lat, lon).

    Assumes monthly data starts in January and contains full years.
    """
    ntime = data.shape[0]
    nyear = ntime // 12
    data = data[:nyear * 12]
    data = data.reshape(nyear, 12, data.shape[1], data.shape[2])

    if season == "JJA":
        # June, July, August
        return data[:, 5:8, :, :]

    elif season == "DJF":
        # December of year i, January+February of year i+1
        return np.stack([
            data[:-1, 11, :, :],   # Dec
            data[1:,   0, :, :],   # Jan
            data[1:,   1, :, :]    # Feb
        ], axis=1)

    else:
        raise ValueError("season must be 'JJA' or 'DJF'")


def seasonal_mean_and_eke(u, uu, v, vv, season="JJA"):
    """
    From monthly U, UU, V, VV arrays, compute:
      - seasonal mean wind speed climatology
      - seasonal mean U climatology
      - seasonal mean V climatology
      - EKE climatology based on monthly anomalies within the selected season

    Inputs:
      u, uu, v, vv : arrays with shape (time, lat, lon)

    Returns:
      vel_speed_mean : (lat, lon)
      u_mean         : (lat, lon)
      v_mean         : (lat, lon)
      EKE_mean       : (lat, lon)
    """
    u_season  = get_season(u,  season=season)
    uu_season = get_season(uu, season=season)
    v_season  = get_season(v,  season=season)
    vv_season = get_season(vv, season=season)

    nseason, _, nlat, nlon = u_season.shape

    # Flatten seasonal months back to one monthly time axis for anomaly/EKE calculation
    u_flat = u_season.reshape(-1, nlat, nlon)
    v_flat = v_season.reshape(-1, nlat, nlon)

    Uprime = u_flat - np.mean(u_flat, axis=0)
    Vprime = v_flat - np.mean(v_flat, axis=0)

    EKE = 0.5 * (Uprime**2 + Vprime**2)
    EKE_mean = np.mean(EKE, axis=0)

    # Mean seasonal wind speed and mean wind vectors
    vel_speed_mean = np.mean(np.sqrt(uu_season + vv_season), axis=(0, 1))
    u_mean = np.mean(u_season, axis=(0, 1))
    v_mean = np.mean(v_season, axis=(0, 1))

    return vel_speed_mean, u_mean, v_mean, EKE_mean


def load_jet_file(filepath):
    """
    Load U, UU, V, VV from NetCDF jet file.
    """
    fh = netcdf.Dataset(filepath, 'r')

    out = {
        "time": fh.variables['time'][:],
        "lon":  fh.variables['lon'][:],
        "lat":  fh.variables['lat'][:],
        "U":    fh.variables['U'][:],
        "UU":   fh.variables['UU'][:],
        "V":    fh.variables['V'][:],
        "VV":   fh.variables['VV'][:]}

    fh.close()

    return out

def apply_periodic_to_jet_dict(data):
    """
    Apply PeriodicBoundaries3D to U, V, UU, VV in a loaded jet-file dict.
    """
    lon = data["lon"]
    lat = data["lat"]

    lon_new, U_new  = PeriodicBoundaries3D(lon, lat, data["U"])
    _,       V_new  = PeriodicBoundaries3D(lon, lat, data["V"])
    _,       UU_new = PeriodicBoundaries3D(lon, lat, data["UU"])
    _,       VV_new = PeriodicBoundaries3D(lon, lat, data["VV"])

    data_out = data.copy()
    data_out["lon"] = lon_new
    data_out["U"]   = U_new
    data_out["V"]   = V_new
    data_out["UU"]  = UU_new
    data_out["VV"]  = VV_new

    return data_out


#%% PI18 comparison (3800 - 600)

data_3800 = load_jet_file(directory_data + 'Jet_200hPa_Southern_Ocean_month_1-12_branch3800_year_4199_4300.nc')
data_3800 = apply_periodic_to_jet_dict(data_3800)

data_600 = load_jet_file(directory_data + 'Jet_200hPa_Southern_Ocean_month_1-12_branch600_year_999_1100.nc')
data_600 = apply_periodic_to_jet_dict(data_600)

lon_vel = data_3800["lon"]
lat_vel = data_3800["lat"]

#%%

#Compute seasonal climatologies and EKE
if season == 'JJA' or 'DJF':
    print(season)
    vel_speed_3800, u_mean_3800, v_mean_3800, EKE_mean_3800 = seasonal_mean_and_eke(
    data_3800["U"], data_3800["UU"], data_3800["V"], data_3800["VV"], season=season)

    vel_speed_600, u_mean_600, v_mean_600, EKE_mean_600 = seasonal_mean_and_eke(
    data_600["U"], data_600["UU"], data_600["V"], data_600["VV"], season=season) 
#Else just use the monthly data
elif season == 'monthly':
    print('Monthly data')
    # Mean wind speed and mean wind vectors
    vel_speed_mean_3800 = np.mean(np.sqrt(data_3800["UU"] + data_3800["VV"]), axis=(0, 1))
    u_mean_3800 = np.mean(data_3800["U"], axis=(0, 1))
    v_mean_3800 = np.mean(data_3800["V"], axis=(0, 1))

    vel_speed_mean_600 = np.mean(np.sqrt(data_600["UU"] + data_600["VV"]), axis=(0, 1))
    u_mean_600 = np.mean(data_600["U"], axis=(0, 1))
    v_mean_600 = np.mean(data_600["V"], axis=(0, 1))

# Differences: PI_off_18 - PI_on_18
vel_speed_plot = vel_speed_3800 - vel_speed_600
u_vel_plot     = u_mean_3800 - u_mean_600
v_vel_plot     = v_mean_3800 - v_mean_600
#EKE_plot       = EKE_mean_3800 - EKE_mean_600


#%% LOAD DATA: PI45 comparison (2900 - 1500)

data_2900 = load_jet_file(directory_data + 'Jet_200hPa_Southern_Ocean_month_1-12_branch2900_year_3299_3400.nc')
data_2900 = apply_periodic_to_jet_dict(data_2900)

data_1500 = load_jet_file(directory_data + 'Jet_200hPa_Southern_Ocean_month_1-12_branch1500_year_1899_2000.nc')
data_1500 = apply_periodic_to_jet_dict(data_1500)

# Compute seasonal climatologies and EKE
vel_speed_2900, u_mean_2900, v_mean_2900, EKE_mean_2900 = seasonal_mean_and_eke(data_2900["U"], data_2900["UU"], data_2900["V"], data_2900["VV"], season=season)
vel_speed_1500, u_mean_1500, v_mean_1500, EKE_mean_1500 = seasonal_mean_and_eke(data_1500["U"], data_1500["UU"], data_1500["V"], data_1500["VV"], season=season)

# Differences: PI_off_45 - PI_on_45
vel_speed_plot_2900_1500 = vel_speed_2900 - vel_speed_1500
u_vel_plot_2900_1500     = u_mean_2900 - u_mean_1500
v_vel_plot_2900_1500     = v_mean_2900 - v_mean_1500
EKE_plot_2900_1500       = EKE_mean_2900 - EKE_mean_1500


# ============================================================
# OPTIONAL: print some basic info
# ============================================================
print(f"Season selected: {season}")
print("PI18 fields:")
print("  vel_speed_plot shape       :", vel_speed_plot.shape)
print("  u_vel_plot shape           :", u_vel_plot.shape)
print("  v_vel_plot shape           :", v_vel_plot.shape)
#print("  EKE_plot shape             :", EKE_plot.shape)

print("PI45 fields:")
print("  vel_speed_plot_2900_1500   :", vel_speed_plot_2900_1500.shape)
print("  u_vel_plot_2900_1500       :", u_vel_plot_2900_1500.shape)
print("  v_vel_plot_2900_1500       :", v_vel_plot_2900_1500.shape)
print("  EKE_plot_2900_1500         :", EKE_plot_2900_1500.shape)


#%%

plt.figure()
plt.plot(lat_vel, np.mean(u_mean_2900, axis=1) - np.mean(u_mean_1500, axis=1))
plt.plot(lat_vel, np.mean(u_mean_3800, axis=1) - np.mean(u_mean_600, axis=1))
plt.axvline(x = -50)
plt.axhline(y=0)
plt.xlim(-70, -30)

#%%

plt.figure()
plt.plot(lat_vel, np.mean(u_mean_2900, axis=1))
plt.plot(lat_vel, np.mean(u_mean_1500, axis=1))
plt.plot(lat_vel, np.mean(u_mean_600, axis=1))
plt.plot(lat_vel, np.mean(u_mean_3800, axis=1))
plt.axhline(y=0)

#%%-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

CS      = ax.contourf(lon_vel, lat_vel, u_mean_600, levels = np.arange(-2, 2.1, 0.1), extend = 'both', cmap = 'PuOr_r', transform=ccrs.PlateCarree())

divider = make_axes_locatable(ax)
ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)

cbar    = fig.colorbar(CS, ticks = np.arange(-2, 2.01, 0.5), cax=ax_cb)
cbar.set_label('Wind speed difference (m s$^{-1}$)')

scale_arrow	= 4
Q = ax.quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot[::scale_arrow, ::scale_arrow], v_vel_plot[::scale_arrow, ::scale_arrow], scale = 100, transform=ccrs.PlateCarree())

#qk = ax.quiverkey(Q, 0.17, 0.10, 10, '10 m s$^{-1}$', labelpos = 'S', coordinates='figure')

#ax.plot([-45, -45], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')
#ax.plot([15, 15], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_extent([-180, 180, -90, -30.001], ccrs.PlateCarree())
ax.coastlines('110m')
ax.add_feature(cfeature.LAND, zorder=0)

ax.set_title('d) Difference 200 hPa velocities (monthly, $\overline{F_H}$ = 0.18Sv)')
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_4_CD_B.pdf')
plt.show()

#-----------------------------------------------------------------------------------------

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

CS      = ax.contourf(lon_vel, lat_vel, vel_speed_plot_2900_1500, levels = np.arange(-2, 2.1, 0.1), extend = 'both', cmap = 'PuOr_r', transform=ccrs.PlateCarree())

divider = make_axes_locatable(ax)
ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)

cbar    = fig.colorbar(CS, ticks = np.arange(-2, 2.01, 1), cax=ax_cb)
cbar.set_label('Wind speed difference (m s$^{-1}$)')

scale_arrow	= 4
Q = ax.quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], v_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], scale = 100, transform=ccrs.PlateCarree())

#qk = ax.quiverkey(Q, 0.17, 0.10, 10, '10 m s$^{-1}$', labelpos = 'S', coordinates='figure')

#ax.plot([-45, -45], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')
#ax.plot([15, 15], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_extent([-180, 180, -90, -30.001], ccrs.PlateCarree())
ax.coastlines('110m')
ax.add_feature(cfeature.LAND, zorder=0)

ax.set_title('d) Difference 200 hPa velocities (monthly, $\overline{F_H}$ = 0.45Sv)')
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_4_CD_B.pdf')
plt.show()


#%% Be aware of wrong sings of EOFs, so alwasy double check this

fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

for ax in axs.flat:
    ax.coastlines()
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

c1 = axs[0,0].contourf(lon, lat, -EOF_SLP_E1*np.std(PC_E1[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-4,4,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[0,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-4, 4.1, 1)) 
cbar.set_label('[hPa]')
axs[0,0].set_title('a) EOF1 pattern - '+str(season)+' SLP (PI$^{\mathrm{on}}_{18}$)')
gl = axs[0,0].gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)

gl.xlines = False
gl.top_labels = False
gl.bottom_labels = False

levels = [-2, -1, 0, 4, 7]

c2 = axs[0,1].contourf(lon_vel, lat_vel, vel_speed_600, transform=ccrs.PlateCarree(), levels = np.arange(10, 45.1, 0.5), extend = 'both', cmap = 'Spectral_r')
cbar = fig.colorbar(c2, ax=axs[0,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) 200 hPa velocities - '+str(season)+' (PI$^{\mathrm{on}}_{18}$)')
axs[0,1].gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)

cbar.set_label('Wind speed [m s$^{-1}$]')

scale_arrow	= 5
Q = axs[0,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_mean_600[::scale_arrow, ::scale_arrow], v_mean_600[::scale_arrow, ::scale_arrow], scale = 300, transform=ccrs.PlateCarree())
axs[0,1].quiverkey(Q,X=0.1, Y=-.07,U=30.0,label='30 m/s',labelpos='E', fontproperties={'size': 11})

c1 = axs[1,0].contourf(lon, lat, -EOF_SLP_E4*np.std(PC_E4[0,:]) - -EOF_SLP_E1*np.std(PC_E1[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-1.5,1.5,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[1,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-1.5, 1.51, 0.5)) 
cbar.set_label('[hPa]')
axs[1,0].set_title('c) Difference EOF1 pattern - '+str(season)+' SLP (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
gl = axs[1,0].gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)

gl.xlines = False
gl.top_labels = False
gl.bottom_labels = False

levels = [-2, -1, 0, 4, 7]

c2 = axs[1,1].contourf(lon_vel, lat_vel, vel_speed_plot, transform=ccrs.PlateCarree(), levels = np.arange(-3, 3.1, 0.5), extend = 'both', cmap = 'PuOr_r')
cbar = fig.colorbar(c2, ax=axs[1,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Difference 200 hPa velocities - '+str(season)+' (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
axs[1,1].gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)
cbar.set_label('Wind speed difference [m s$^{-1}$]')

scale_arrow	= 4
Q = axs[1,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot[::scale_arrow, ::scale_arrow], v_vel_plot[::scale_arrow, ::scale_arrow], scale = 30, transform=ccrs.PlateCarree())
axs[1,1].quiverkey(Q,X=0.1, Y=-.07,U=3.0,label='3 m/s',labelpos='E', fontproperties={'size': 11})

plt.tight_layout()
plt.subplots_adjust(left=0.03)
plt.savefig(directory_figures +'Figure_4_CD_'+str(season)+'_018.pdf')
plt.show()

#%%

fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

for ax in axs.flat:
    ax.coastlines()
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

c1 = axs[0,0].contourf(lon, lat, EOF_SLP_E2*np.std(PC_E2[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-4,4,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[0,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-4, 4.1, 1)) 
cbar.set_label('[hPa]')
axs[0,0].set_title('a) EOF1 pattern - '+str(season)+' SLP (PI$^{\mathrm{on}}_{45}$)')
gl = axs[0,0].gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)

gl.xlines = False
gl.top_labels = False
gl.bottom_labels = False

levels = [-2, -1, 0, 4, 7]

c2 = axs[0,1].contourf(lon_vel, lat_vel, vel_speed_1500, transform=ccrs.PlateCarree(), levels = np.arange(10, 45.1, 0.5), extend = 'both', cmap = 'Spectral_r')
cbar = fig.colorbar(c2, ax=axs[0,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) 200 hPa velocities - '+str(season)+' (PI$^{\mathrm{on}}_{45}$)')
axs[0,1].gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)
cbar.set_label('Wind speed [m s$^{-1}$]')

scale_arrow	= 5
Q = axs[0,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_mean_1500[::scale_arrow, ::scale_arrow], v_mean_1500[::scale_arrow, ::scale_arrow], scale = 300, transform=ccrs.PlateCarree())
axs[0,1].quiverkey(Q,X=0.1, Y=-.07,U=30.0,label='30 m/s',labelpos='E', fontproperties={'size': 11})


c1 = axs[1,0].contourf(lon, lat, EOF_SLP_E3*np.std(PC_E3[0,:]) - EOF_SLP_E2*np.std(PC_E2[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-1.5,1.5,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[1,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-1.5, 1.51, 0.5)) 
cbar.set_label('[hPa]')
axs[1,0].set_title('c) Difference EOF1 pattern - '+str(season)+' SLP (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
gl = axs[1,0].gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)

gl.xlines = False
gl.top_labels = False
gl.bottom_labels = False

levels = [-2, -1, 0, 4, 7]

c2 = axs[1,1].contourf(lon_vel, lat_vel, vel_speed_plot_2900_1500, transform=ccrs.PlateCarree(), levels = np.arange(-3, 3.1, 0.5), extend = 'both', cmap = 'PuOr_r')
cbar = fig.colorbar(c2, ax=axs[1,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Difference 200 hPa velocities - '+str(season)+' (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
axs[1,1].gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)
cbar.set_label('Wind speed difference [m s$^{-1}$]')

scale_arrow	= 4
Q = axs[1,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], v_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], scale = 30, transform=ccrs.PlateCarree())
axs[1,1].quiverkey(Q,X=0.1, Y=-.07,U=3.0,label='3 m/s',labelpos='E', fontproperties={'size': 11})

plt.tight_layout()
plt.subplots_adjust(left=0.02)
plt.savefig(directory_figures +'Figure_SAM_SM_'+str(season)+'_045.pdf')
plt.show()

#%% Determine statistics of the PCs to see if something changes there

def pc_stats_ma(pc):
    pc = ma.array(pc).compressed()  # drop masked
    return dict(
        mean=float(pc.mean()),
        std=float(pc.std(ddof=1)),
        p05=float(np.percentile(pc, 5)),
        p50=float(np.percentile(pc, 50)),
        p95=float(np.percentile(pc, 95)),
        skew=float(stats.skew(pc, bias=False)),
    )

def compare_pcs(pc_on, pc_off):
    x = ma.array(pc_on).compressed()
    y = ma.array(pc_off).compressed()
    return dict(
        welch_t=stats.ttest_ind(y, x, equal_var=False),   # mean shift
        levene=stats.levene(y, x),                        # variance change
        ks=stats.ks_2samp(y, x),                          # distro change
        dmean=float(y.mean()-x.mean()),
        dstd=float(y.std(ddof=1)-x.std(ddof=1)),
    )

# example
s_on  = pc_stats_ma(PC_E1[0,:])
s_off = pc_stats_ma(PC_E4[0,:])
print("E1 on:", s_on)
print("E4 off:", s_off)
print(compare_pcs(PC_E1[0,:], PC_E4[0,:]))

s_on  = pc_stats_ma(PC_E2[0,:])
s_off = pc_stats_ma(PC_E3[0,:])
print("E2 on:", s_on)
print("E3 off:", s_off)
print(compare_pcs(PC_E2[0,:], PC_E3[0,:]))

#%%

pos_fraction = np.mean(PC_E1[0,:] > 0)
neg_fraction = np.mean(PC_E1[0,:] < 0)

print(pos_fraction, neg_fraction)

#%%

def extreme_sam_fraction(pc):
    pc = np.asarray(pc)
    sigma = np.std(pc)

    frac_pos_1sigma = np.mean(pc > sigma)
    frac_neg_1sigma = np.mean(pc < -sigma)

    return frac_pos_1sigma, frac_neg_1sigma

pos1, neg1 = extreme_sam_fraction(PC_E1[0,:])
print(pos1, neg1)


#%%

fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': ccrs.SouthPolarStereo()})

for ax in axs.flat:  
    ax.coastlines()
    ax.set_extent([-185, 190, -90, -30], crs=ccrs.PlateCarree())  

    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', color='gray', alpha=0.7)
    gl.top_labels = False  
    gl.right_labels = False  
    gl.xlabel_style = {'size': 10, 'color': 'black'}  
    gl.ylabel_style = {'size': 10, 'color': 'black'}  

data = [
    (EOF_E1[1,:,:] * np.std((PC_E1[1,:])) - -EOF_E4[1,:,:] * np.std((PC_E4[1,:])), 'a) EOF2 monthly SLP - PI$^{on}_{18}$ (var.ex. = '+str(int(VAR_E1[1]*100))+'%)', np.linspace(-2, 2, 21)),
    (-EOF_E2[1,:,:] * np.std((PC_E2[1,:])) - EOF_E3[1,:,:] * np.std((PC_E3[1,:])), 'b) EOF2 monthly SLP - PI$^{off}_{18}$ (var.ex. = '+str(int(VAR_E4[1]*100))+'%)', np.linspace(-3, 3, 21)),
    (-EOF_E2[1,:,:] * np.std((PC_E2[1,:])) - EOF_E3[1,:,:] * np.std((PC_E3[1,:])), 'c) EOF2 monthly SLP - PI$^{on}_{45}$ (var.ex. = '+str(int(VAR_E2[1]*100))+'%)', np.linspace(-7, 7, 21)),
    (EOF_E3[1,:,:] * np.std((PC_E3[1,:])), 'd) EOF2 monthly SLP - PI$^{off}_{45}$ (var.ex. = '+str(int(VAR_E3[1]*100))+'%)', np.linspace(-7, 7, 21))
]

for ax, (EOF, title, levels) in zip(axs.flat, data):
    contour = ax.contourf(
        lon, lat, EOF, transform=ccrs.PlateCarree(),
        levels=levels, cmap='RdBu_r', extend='both'
    )
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.8)
    #cbar.set_label('EOF Value', fontsize=10)
    ax.set_title(title, fontsize=12)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_SAM_EOF_SM.pdf')
plt.show() 


# %%
