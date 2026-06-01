#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:45:46 2025

@author: 6008399

Mean SST and wind stress over equatorial Pacific

"""
#%%
from pylab import *
import numpy
import datetime
import time
import glob, os
import math
import netCDF4 as netcdf1
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
from scipy.signal.windows import dpss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.patches import Rectangle

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%% Read in data

lat1, lat2 = 110, 260

#fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_Indian_year_999-1100_month_1-12_branch600.nc', 'r')
fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_999-1100_month_1-12_branch600.nc', 'r')

time_month_E1       = fh.variables['time_month'][0:100*12] #Model years
time_E1       = fh.variables['time'][0:100] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E1        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E1              = fh.variables['SST'][0:100, lat1:lat2, :]
TAUX_month_E1       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E1             = fh.variables['TAUX'][0:100, lat1:lat2, :]

fh.close()


#fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_Indian_year_1899-2000_month_1-12_branch1500.nc', 'r')
fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_1899-2000_month_1-12_branch1500.nc', 'r')

time_month_E2       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E2        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E2              = fh.variables['SST'][0:100, lat1:lat2, :]
TAUX_month_E2       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E2             = fh.variables['TAUX'][0:100, lat1:lat2, :]

fh.close()

#fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_Indian_year_3299-3400_month_1-12_branch2900.nc', 'r')
fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_3299-3400_month_1-12_branch2900.nc', 'r')

time_month_E3       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E3        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E3              = fh.variables['SST'][0:100, lat1:lat2, :]
TAUX_month_E3       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E3             = fh.variables['TAUX'][0:100, lat1:lat2, :]

fh.close()

#fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_Indian_year_4199-4300_month_1-12_branch3800.nc', 'r')
fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_4199-4300_month_1-12_branch3800.nc', 'r')
time_month_E4       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E4        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E4              = fh.variables['SST'][0:100, lat1:lat2, :]
TAUX_month_E4       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E4             = fh.variables['TAUX'][0:100, lat1:lat2, :]

fh.close()

#%%

fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_999-1100_month_12-14_branch600.nc', 'r')

time_month_E1_DJF   = fh.variables['time_month'][:] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E1_DJF    = fh.variables['SST_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E1_DJF          = fh.variables['SST'][:, lat1:lat2, :]
TAUX_month_E1_DJF   = fh.variables['TAUX_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E1_DJF         = fh.variables['TAUX'][:, lat1:lat2, :]

fh.close()


fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_1899-2000_month_12-14_branch1500.nc', 'r')

#time_month_E2       = fh.variables['time_month'][:] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E2_DJF    = fh.variables['SST_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E2_DJF          = fh.variables['SST'][:, lat1:lat2, :]
TAUX_month_E2_DJF   = fh.variables['TAUX_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E2_DJF         = fh.variables['TAUX'][:, lat1:lat2, :]

fh.close()

fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_3299-3400_month_12-14_branch2900.nc', 'r')

#time_month_E3       = fh.variables['time_month'][:] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E3_DJF    = fh.variables['SST_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E3_DJF          = fh.variables['SST'][:, lat1:lat2, :]
TAUX_month_E3_DJF   = fh.variables['TAUX_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E3_DJF         = fh.variables['TAUX'][:, lat1:lat2, :]

fh.close()

fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_4199-4300_month_12-14_branch3800.nc', 'r')

#time_month_E4       = fh.variables['time_month'][:] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E4_DJF    = fh.variables['SST_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E4_DJF          = fh.variables['SST'][:, lat1:lat2, :]
TAUX_month_E4_DJF   = fh.variables['TAUX_month'][:, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E4_DJF         = fh.variables['TAUX'][:, lat1:lat2, :]

fh.close()


#%% Take only spring data and average over that

SST_E1_MAM  = ma.masked_all((int(len(time_month_E1)/12), len(lat), len(lon[0])))
TAUX_E1_MAM = ma.masked_all((int(len(time_month_E1)/12), len(lat), len(lon[0])))

for i in range(int(len(time_month_E1)/12)-1):
    SST_E1_MAM[i,:,:]   = np.mean(SST_month_E1[i + 2: i + 5], axis=0)
    TAUX_E1_MAM[i,:,:]  = np.mean(TAUX_month_E1[i + 2: i + 5], axis=0)

SST_E2_MAM  = ma.masked_all((int(len(time_month_E2)/12), len(lat), len(lon[0])))
TAUX_E2_MAM = ma.masked_all((int(len(time_month_E2)/12), len(lat), len(lon[0])))

for i in range(int(len(time_month_E2)/12)-1):
    SST_E2_MAM[i,:,:]   = np.mean(SST_month_E2[i + 2: i + 5], axis=0)
    TAUX_E2_MAM[i,:,:]  = np.mean(TAUX_month_E2[i + 2: i + 5], axis=0)
    
SST_E3_MAM  = ma.masked_all((int(len(time_month_E3)/12), len(lat), len(lon[0])))
TAUX_E3_MAM = ma.masked_all((int(len(time_month_E3)/12), len(lat), len(lon[0])))

for i in range(int(len(time_month_E3)/12)-1):
    SST_E3_MAM[i,:,:]   = np.mean(SST_month_E3[i + 2: i + 5], axis=0)
    TAUX_E3_MAM[i,:,:]  = np.mean(TAUX_month_E3[i + 2: i + 5], axis=0)

SST_E4_MAM  = ma.masked_all((int(len(time_month_E4)/12), len(lat), len(lon[0])))
TAUX_E4_MAM = ma.masked_all((int(len(time_month_E4)/12), len(lat), len(lon[0])))

for i in range(int(len(time_month_E4)/12)-1):
    SST_E4_MAM[i,:,:]   = np.mean(SST_month_E4[i + 2: i + 5], axis=0)
    TAUX_E4_MAM[i,:,:]  = np.mean(TAUX_month_E4[i + 2: i + 5], axis=0)
    
#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('Winter averaged Pacific SST', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E1_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(21,30,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) AMOC on $F_H$ = 0.18Sv')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E2_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(21,30,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) AMOC on $F_H$ = 0.45Sv')
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.mean(SST_E3_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(21,30,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) AMOC off $F_H$ = 0.18Sv')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.mean(SST_E4_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(21,30,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) AMOC off $F_H$ = 0.45Sv')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Adjust the layout
from matplotlib.patches import Rectangle

# Draw Nino3.4 box (5S-5N, 170W-120W). Handle lon in either -180..180 or 0..360
lon_vals = lon.mean(axis=0) if getattr(lon, 'ndim', 1) == 2 else lon
if np.nanmax(lon_vals) > 180:
    lon_box_min, lon_box_max = 360 - 170, 360 - 120  # 190, 240
else:
    lon_box_min, lon_box_max = -170, -120
lat_box_min, lat_box_max = -5, 5
width = lon_box_max - lon_box_min
height = lat_box_max - lat_box_min
for ax in axs.flat:
    rect = Rectangle((lon_box_min, lat_box_min), width, height,
                     linewidth=2, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)

plt.tight_layout()
plt.savefig(directory_figures +'ENSO_SST_Pacific_CESM_branches.pdf')
plt.show() 


#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E4_DJF, axis=0) - np.mean(SST_E1_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) SST difference (DJF, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(SST_E1_DJF[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(SST_E1_DJF[:, lat_i, lon_i], SST_E4_DJF[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0,0].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E3_DJF, axis=0) - np.mean(SST_E2_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) SST difference (DJF, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(SST_E2_DJF[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(SST_E2_DJF[:, lat_i, lon_i], SST_E3_DJF[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0,1].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

c1 = axs[1,0].contourf(lon, lat, np.mean(TAUX_E4_DJF, axis=0) - np.mean(TAUX_E1_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
#axs[1,0].axhline(y=5)
#axs[1,0].axhline(y=-5)
axs[1,0].set_title('c) Zonal windstress difference (DJF, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(TAUX_E1_DJF[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(TAUX_E1_DJF[:, lat_i, lon_i], TAUX_E4_DJF[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1,0].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

c2 = axs[1,1].contourf(lon, lat, np.mean(TAUX_E3_DJF, axis=0) - np.mean(TAUX_E2_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
#axs[1,1].axhline(y=5)
#axs[1,1].axhline(y=-5)
axs[1,1].set_title('d) Zonal windstress difference (DJF, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(TAUX_E2_DJF[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(TAUX_E2_DJF[:, lat_i, lon_i], TAUX_E3_DJF[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1,1].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

gl = axs[0,0].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())
gl = axs[0,1].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())
gl = axs[1,0].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())
gl = axs[1,1].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'winter_ENSO_TAUX_SST_differences_Pacific_CESM_branches.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E1, axis=0), transform=ccrs.PlateCarree())#, levels = np.linspace(-2,2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) SST difference (annual, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(SST_E1[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(SST_E1[:, lat_i, lon_i], SST_E4[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0,0].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E3, axis=0) - np.mean(SST_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) SST difference (annual, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(SST_E2[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(SST_E2[:, lat_i, lon_i], SST_E3[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0,1].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

c1 = axs[1,0].contourf(lon, lat, np.mean(TAUX_E4, axis=0) - np.mean(TAUX_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
#axs[1,0].axhline(y=5)
#axs[1,0].axhline(y=-5)
axs[1,0].set_title('c) Zonal wind stress difference (annual, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

skip_y = 10
skip_x = 15

tau_on = np.mean(TAUX_E1, axis=0)

Q1 = axs[1,0].quiver(
    lon[::skip_y, ::skip_x],
    lat[::skip_y, ::skip_x],
    tau_on[::skip_y, ::skip_x],
    np.zeros_like(tau_on[::skip_y, ::skip_x]),
    transform=ccrs.PlateCarree(),
    color='grey',
    scale=10)

axs[1,0].quiverkey(
    Q1, 0.05, -1.1, 0.5,
    r'0.5 N m$^{-2}$',
    labelpos='E')

# for lat_i in range(0, len(lat), 10):
#     for lon_i in range(0, len(lon[0]), 5):
#         if ma.is_masked(TAUX_E1_DJF[0, lat_i, lon_i]):
#             continue  # Skip if land
#         p_value = Welch(TAUX_E1_DJF[:, lat_i, lon_i], TAUX_E4_DJF[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             axs[1,0].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

c2 = axs[1,1].contourf(lon, lat, np.mean(TAUX_E3, axis=0) - np.mean(TAUX_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
#axs[1,1].axhline(y=5)
#axs[1,1].axhline(y=-5)
axs[1,1].set_title('d) Zonal wind stress difference (annual, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

skip_y = 10
skip_x = 17

tau_on = np.mean(TAUX_E2, axis=0)

Q2 = axs[1,1].quiver(
    lon[::skip_y, ::skip_x],
    lat[::skip_y, ::skip_x],
    tau_on[::skip_y, ::skip_x],
    np.zeros_like(tau_on[::skip_y, ::skip_x]),
    transform=ccrs.PlateCarree(),
    color='grey',
    scale=10)

axs[1,1].quiverkey(
    Q2, 0.05, -1.1, 0.5,
    r'0.5 N m$^{-2}$',
    labelpos='E')

# for lat_i in range(0, len(lat), 10):
#     for lon_i in range(0, len(lon[0]), 5):
#         if ma.is_masked(TAUX_E2_DJF[0, lat_i, lon_i]):
#             continue  # Skip if land
#         p_value = Welch(TAUX_E2_DJF[:, lat_i, lon_i], TAUX_E3_DJF[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             axs[1,1].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

gl = axs[0,0].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())
gl = axs[0,1].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())
gl = axs[1,0].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())
gl = axs[1,1].gridlines(draw_labels=False,xlocs=np.arange(-190,170,30),ylocs=np.arange(-40,81,20),crs=ccrs.PlateCarree())

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'annual_ENSO_TAUX_SST_differences_Pacific_CESM_branches.pdf')
plt.show() 

#%%

plt.figure(figsize=(8, 4))
plt.contourf(lon, lat, np.mean(SST_E1, axis=0), levels = np.linspace(20,30,21), cmap='Spectral_r', extend='both')

#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(20,30,21), cmap='Spectral_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) SST (annual, PI$^{\mathrm{on}}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(20,30,21), cmap='Spectral_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) SST (annual, PI$^{\mathrm{on}}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.mean(SST_E4, axis=0) - np.mean(SST_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) SST difference (annual, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.mean(SST_E3, axis=0) - np.mean(SST_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) SST difference (annual, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Draw Nino3.4 box (5S-5N, 170W-120W). Handle lon in either -180..180 or 0..360
lon_vals = lon.mean(axis=0) if getattr(lon, 'ndim', 1) == 2 else lon
if np.nanmax(lon_vals) > 180:
    lon_box_min, lon_box_max = 360 - 170, 360 - 120  # 190, 240
else:
    lon_box_min, lon_box_max = -170, -120
lat_box_min, lat_box_max = -5, 5
width = lon_box_max - lon_box_min
height = lat_box_max - lat_box_min
for ax in axs.flat:
    rect = Rectangle((lon_box_min, lat_box_min), width, height,
                     linewidth=2, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'annual_ENSO_SST_differences_Pacific_CESM_branches.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(TAUX_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) Zonal wind stress (annual, PI$^{\mathrm{on}}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.mean(TAUX_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) Zonal wind stress (annual, PI$^{\mathrm{on}}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.mean(TAUX_E4, axis=0) - np.mean(TAUX_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.2,0.2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) Zonal wind stress difference (annual, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.mean(TAUX_E3, axis=0) - np.mean(TAUX_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.2,0.2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Zonal wind stress difference (annual, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'annual_ENSO_TAUx_differences_Pacific_CESM_branches.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('Winter averaged SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E4_DJF, axis=0) - np.mean(SST_E1_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) SST difference ($F_H$ = 0.18Sv)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E3_DJF, axis=0) - np.mean(SST_E2_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='seismic', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('SST difference (AMOC off minus on)')
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.mean(TAUX_E4_DJF, axis=0) - np.mean(TAUX_E1_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) TAUx difference ($F_H$ = 0.18Sv)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.mean(TAUX_E3_DJF, axis=0) - np.mean(TAUX_E2_DJF, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) TAUx difference ($F_H$ = 0.45Sv)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'winter_ENSO_TAUX_SST_differences_Pacific_CESM_branches.pdf')
plt.show() 


#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('Spring averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E4_MAM, axis=0) - np.mean(SST_E1_MAM, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) SST difference ($F_H$ = 0.18Sv)')
#axs[0,0].quiver(np.mean(TAUX,E1_MAM, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E3_MAM, axis=0) - np.mean(SST_E2_MAM, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-2,2,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) SST difference ($F_H$ = 0.45Sv)')
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.mean(TAUX_E4_MAM, axis=0) - np.mean(TAUX_E1_MAM, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) TAUx difference ($F_H$ = 0.18Sv)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.mean(TAUX_E3_MAM, axis=0) - np.mean(TAUX_E2_MAM, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.3,0.3,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) TAUx difference ($F_H$ = 0.45Sv)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'spring_ENSO_TAUX_SST_differences_Pacific_CESM_branches.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.mean(SST_E4, axis=0) - np.mean(SST_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) SST difference (annual, PI$^{off}_{18}$ - PI$^{on}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.mean(SST_E3, axis=0) - np.mean(SST_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) SST difference (annual, PI$^{off}_{45}$ - PI$^{on}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.mean(TAUX_E4, axis=0) - np.mean(TAUX_E1, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.2,0.2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
#axs[1,0].axhline(y=5)
#axs[1,0].axhline(y=-5)
axs[1,0].set_title('c) Zonal windstress difference (annual, PI$^{off}_{18}$ - PI$^{on}_{18}$)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.mean(TAUX_E3, axis=0) - np.mean(TAUX_E2, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.2,0.2,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
#axs[1,1].axhline(y=5)
#axs[1,1].axhline(y=-5)
axs[1,1].set_title('d) Zonal windstress difference (annual, PI$^{off}_{45}$ - PI$^{on}_{45}$)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'annual_ENSO_TAUX_SST_differences_Pacific_CESM_branches.pdf')
plt.show() 

#%% Standard deviation differences using monthly data 

def TrendRemover(time, data, trend_type):
    # NOTE: modifies a copy if you pass .copy()
    rank = np.polyfit(time, data, trend_type)
    fitting = np.zeros(len(time))
    for i in range(len(rank)):
        fitting += rank[i] * (time ** (len(rank) - 1 - i))
    return data - fitting

#First, we should check if the standard deviation of SST is different between the two branches, which may indicate a change in ENSO variability. We can do this by calculating the standard deviation of monthly SST for each branch and then plotting the difference.
def remove_monthly_climatology(x):
    """
    Remove the mean monthly cycle from x with shape (ntime, nlat, nlon).
    Assumes x is monthly and starts at January. Works with numpy masked arrays
    or arrays containing NaNs.
    Returns a copy with the monthly climatology removed.
    """
    x = x.copy()
    ntime = x.shape[0]
    is_masked = isinstance(x, np.ma.MaskedArray)
    for m in range(12):
        idx = np.arange(m, ntime, 12)
        if is_masked:
            clim = np.ma.mean(x[idx, ...], axis=0)
            x[idx, ...] = x[idx, ...] - clim
        else:
            clim = np.nanmean(x[idx, ...], axis=0)
            x[idx, ...] = x[idx, ...] - clim
    return x

#Remove seasonal climatology from monthly data
SST_month_E1_anom = remove_monthly_climatology(SST_month_E1)
SST_month_E2_anom = remove_monthly_climatology(SST_month_E2)
SST_month_E3_anom = remove_monthly_climatology(SST_month_E3)
SST_month_E4_anom = remove_monthly_climatology(SST_month_E4)

SST_month_E1_DJF_anom = remove_monthly_climatology(SST_month_E1_DJF)
SST_month_E2_DJF_anom = remove_monthly_climatology(SST_month_E2_DJF)
SST_month_E3_DJF_anom = remove_monthly_climatology(SST_month_E3_DJF)
SST_month_E4_DJF_anom = remove_monthly_climatology(SST_month_E4_DJF)

TAUX_month_E1_DJF_anom = remove_monthly_climatology(TAUX_month_E1_DJF)
TAUX_month_E2_DJF_anom = remove_monthly_climatology(TAUX_month_E2_DJF)
TAUX_month_E3_DJF_anom = remove_monthly_climatology(TAUX_month_E3_DJF)
TAUX_month_E4_DJF_anom = remove_monthly_climatology(TAUX_month_E4_DJF)

#Remove linear trend
SST_month_E1_anom_DT = ma.masked_all(SST_month_E1_anom.shape)
SST_month_E2_anom_DT = ma.masked_all(SST_month_E2_anom.shape)
SST_month_E3_anom_DT = ma.masked_all(SST_month_E3_anom.shape)
SST_month_E4_anom_DT = ma.masked_all(SST_month_E4_anom.shape)

for lat_i in range(len(lat)):
    for lon_i in range(len(lon[0])):
        SST_month_E1_anom_DT[:, lat_i, lon_i] = TrendRemover(time_month_E1, SST_month_E1_anom[:, lat_i, lon_i], 1)
        SST_month_E2_anom_DT[:, lat_i, lon_i] = TrendRemover(time_month_E2, SST_month_E2_anom[:, lat_i, lon_i], 1)
        SST_month_E3_anom_DT[:, lat_i, lon_i] = TrendRemover(time_month_E3, SST_month_E3_anom[:, lat_i, lon_i], 1)
        SST_month_E4_anom_DT[:, lat_i, lon_i] = TrendRemover(time_month_E4, SST_month_E4_anom[:, lat_i, lon_i], 1)

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.std(SST_month_E4_DJF_anom, axis=0) - np.std(SST_month_E1_DJF_anom, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) std SST (DJF, PI$^{off}_{18}$ - PI$^{on}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

for lat_i in range(0, len(lat), 10):
    for lon_i in range(0, len(lon[0]), 5):
        if ma.is_masked(SST_month_E4_DJF_anom[0, lat_i, lon_i]):
            continue  # Skip if land
        p_value = Welch(SST_month_E4_DJF_anom[:, lat_i, lon_i], SST_month_E1_DJF_anom[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1,1].scatter(lon[lat_i, lon_i], lat[lat_i, lon_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())


c2 = axs[0,1].contourf(lon, lat, np.std(SST_month_E3_DJF_anom, axis=0) - np.std(SST_month_E2_DJF_anom, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) std SST (DJF, PI$^{off}_{45}$ - PI$^{on}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.std(SST_month_E4_anom_DT, axis=0) - np.std(SST_month_E1_anom_DT, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
#axs[1,0].axhline(y=5)
#axs[1,0].axhline(y=-5)
axs[1,0].set_title('c) std SST (monthly, PI$^{off}_{18}$ - PI$^{on}_{18}$)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.std(SST_month_E3_anom_DT, axis=0) - np.std(SST_month_E2_anom_DT, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
#axs[1,1].axhline(y=5)
#axs[1,1].axhline(y=-5)
axs[1,1].set_title('d) std SST (monthly, PI$^{off}_{45}$ - PI$^{on}_{45}$)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

lon_vals = lon.mean(axis=0) if getattr(lon, 'ndim', 1) == 2 else lon
if np.nanmax(lon_vals) > 180:
    lon_box_min, lon_box_max = 360 - 170, 360 - 120  # 190, 240
else:
    lon_box_min, lon_box_max = -170, -120
lat_box_min, lat_box_max = -5, 5
width = lon_box_max - lon_box_min
height = lat_box_max - lat_box_min
for ax in axs.flat:
    rect = Rectangle((lon_box_min, lat_box_min), width, height,
                     linewidth=2, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)


# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'std_SST_DJF_annual_Pacific_CESM_branches.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('Winter averaged Pacific TAUx and SST differences', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, np.std(SST_month_E4_DJF_anom, axis=0) - np.std(SST_month_E1_DJF_anom, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) std SST (DJF, PI$^{off}_{18}$ - PI$^{on}_{18}$)')
#axs[0,0].quiver(np.mean(TAUX,E1_JF, axis=0))
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
#axs[0,0].axhline(y=5)
#axs[0,0].axhline(y=-5)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, np.std(SST_month_E3_DJF_anom, axis=0) - np.std(SST_month_E2_DJF_anom, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-1,1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) std SST (DJF, PI$^{off}_{45}$ - PI$^{on}_{45}$)')
#axs[0,1].axhline(y=5)
#axs[0,1].axhline(y=-5)
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, np.std(TAUX_month_E4_DJF_anom, axis=0) - np.std(TAUX_month_E1_DJF_anom, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.1,0.1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
#axs[1,0].axhline(y=5)
#axs[1,0].axhline(y=-5)
axs[1,0].set_title('c) std zonal wind stress (DJF, PI$^{off}_{18}$ - PI$^{on}_{18}$)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, np.std(TAUX_month_E3_DJF_anom, axis=0) - np.std(TAUX_month_E2_DJF_anom, axis=0), transform=ccrs.PlateCarree(), levels = np.linspace(-0.1,0.1,21), cmap='RdBu_r', extend='both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
#axs[1,1].axhline(y=5)
#axs[1,1].axhline(y=-5)
axs[1,1].set_title('d) std zonal wind stress (DJF, PI$^{off}_{45}$ - PI$^{on}_{45}$)')
axs[1,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

lon_vals = lon.mean(axis=0) if getattr(lon, 'ndim', 1) == 2 else lon
if np.nanmax(lon_vals) > 180:
    lon_box_min, lon_box_max = 360 - 170, 360 - 120  # 190, 240
else:
    lon_box_min, lon_box_max = -170, -120
lat_box_min, lat_box_max = -5, 5
width = lon_box_max - lon_box_min
height = lat_box_max - lat_box_min
for ax in axs.flat:
    rect = Rectangle((lon_box_min, lat_box_min), width, height,
                     linewidth=2, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rect)


# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'std_SST_taux_DJF_Pacific_CESM_branches.pdf')
plt.show() 
#%% Compute the monthly mean as a function of longitude of SST and TAUX for each experiment (5S-5N)

#this is not between 5S and 5N!! See other script of seasonal cycle for that. 

#Compute monthly mean  SST and TAUX for each experiment, averaged between 5S and 5N. This will give us a 2D array of shape (12, nlon) for each variable and experiment, which we can then plot as a contourf to see the seasonal cycle and how it differs between the branches.
nyears_E1 = SST_month_E1.shape[0] // 12
nyears_E2 = SST_month_E2.shape[0] // 12
nyears_E3 = SST_month_E3.shape[0] // 12
nyears_E4 = SST_month_E4.shape[0] // 12

clim_E1 = SST_month_E1.reshape(nyears_E1, 12, SST_month_E1.shape[1], SST_month_E1.shape[2]).mean(axis=0)
clim_E2 = SST_month_E2.reshape(nyears_E2, 12, SST_month_E2.shape[1], SST_month_E2.shape[2]).mean(axis=0)
clim_E3 = SST_month_E3.reshape(nyears_E3, 12, SST_month_E3.shape[1], SST_month_E3.shape[2]).mean(axis=0)
clim_E4 = SST_month_E4.reshape(nyears_E4, 12, SST_month_E4.shape[1], SST_month_E4.shape[2]).mean(axis=0)

clim_E1_taux = TAUX_month_E1.reshape(nyears_E1, 12, TAUX_month_E1.shape[1], TAUX_month_E1.shape[2]).mean(axis=0)
clim_E2_taux = TAUX_month_E2.reshape(nyears_E2, 12, TAUX_month_E2.shape[1], TAUX_month_E2.shape[2]).mean(axis=0)
clim_E3_taux = TAUX_month_E3.reshape(nyears_E3, 12, TAUX_month_E3.shape[1], TAUX_month_E3.shape[2]).mean(axis=0)
clim_E4_taux = TAUX_month_E4.reshape(nyears_E4, 12, TAUX_month_E4.shape[1], TAUX_month_E4.shape[2]).mean(axis=0)

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf = axs[0].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E4, axis=(1)), levels=np.linspace(22,30,21), extend='both', cmap='Spectral_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[0].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[0].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[0])
colorbar.set_label('SST (°C)')
axs[0].set_ylabel('Month')
axs[0].set_title('a) Climatological monthly mean SST (5S-5N) for PI$^{off}_{18}$')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E4_taux, axis=(1)), levels=np.linspace(-0.9,0.1,21), extend='both', cmap='PuOr_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('SST (°C)')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Climatological monthly mean TAUX (5S-5N) for PI$^{off}_{18}$')

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf = axs[0].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E2, axis=(1)), levels=np.linspace(22,30,21), extend='both', cmap='Spectral_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[0].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[0].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[0])
colorbar.set_label('SST (°C)')
axs[0].set_ylabel('Month')
axs[0].set_title('a) Climatological monthly mean SST (5S-5N) for PI$^{on}_{45}$')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E2_taux, axis=(1)), levels=np.linspace(-0.9,0.1,21), extend='both', cmap='PuOr_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('SST (°C)')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Climatological monthly mean TAUX (5S-5N) for PI$^{on}_{45}$')

#%% Annual mean removed

annual_mean_E1 = np.mean(clim_E1, axis=0)   # shape (lat, lon)
clim_anom_E1 = clim_E1 - annual_mean_E1
clim_zonal_E1 = np.mean(clim_anom_E1, axis=1)   # average over latitude

annual_mean_E2 = np.mean(clim_E2, axis=0)   # shape (lat, lon)
clim_anom_E2 = clim_E2 - annual_mean_E2
clim_zonal_E2 = np.mean(clim_anom_E2, axis=1)   # average over latitude

annual_mean_E3 = np.mean(clim_E3, axis=0)   # shape (lat, lon)
clim_anom_E3 = clim_E3 - annual_mean_E3
clim_zonal_E3 = np.mean(clim_anom_E3, axis=1)   # average over latitude

annual_mean_E4 = np.mean(clim_E4, axis=0)   # shape (lat, lon)
clim_anom_E4 = clim_E4 - annual_mean_E4
clim_zonal_E4 = np.mean(clim_anom_E4, axis=1)   # average over latitude

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf = axs[0].contourf(lon[0], np.linspace(1,12, 12), clim_zonal_E1, levels=np.linspace(-2,2,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[0].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[0].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[0])
colorbar.set_label('SST (°C)')
axs[0].set_ylabel('Month')
axs[0].set_title('a) Climatological monthly mean SST (5S-5N) for PI$^{off}_{18}$')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), clim_zonal_E1 - clim_zonal_E4, levels=np.linspace(-1,1,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('SST (°C)')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Climatological monthly mean TAUX (5S-5N) for PI$^{off}_{18}$')

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf = axs[0].contourf(lon[0], np.linspace(1,12, 12), clim_zonal_E2, levels=np.linspace(-2,2,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[0].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[0].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[0])
colorbar.set_label('SST (°C)')
axs[0].set_ylabel('Month')
axs[0].set_title('a) Climatological monthly mean SST (5S-5N) for PI$^{on}_{45}$')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), clim_zonal_E2 - clim_zonal_E3, levels=np.linspace(-1,1,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('SST (°C)')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Climatological monthly mean TAUX (5S-5N) for PI$^{on}_{45}$')



#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf = axs[0].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E4, axis=(1)) - np.mean(clim_E1, axis=(1)), levels=np.linspace(-1,1,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[0].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[0].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[0])
colorbar.set_label('SST [°C]')
axs[0].set_ylabel('Month')
axs[0].set_title('a) Monthly mean SST (5S-5N) (PI$^{off}_{18}$ - PI$^{on}_{18}$)')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E4_taux, axis=(1)) - np.mean(clim_E1_taux, axis=(1)), levels=np.linspace(-0.2,0.2,21), extend='both', cmap='PuOr_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('Zonal wind stress [N/m²]')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Monthly mean TAUX (5S-5N) (PI$^{off}_{18}$ - PI$^{on}_{18}$)')

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf = axs[0].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E3, axis=(1)) - np.mean(clim_E2, axis=(1)), levels=np.linspace(-1,1,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[0].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[0].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[0])
colorbar.set_label('SST [°C]')
axs[0].set_ylabel('Month')
axs[0].set_title('a) Monthly mean SST (5S-5N) (PI$^{off}_{45}$ - PI$^{on}_{45}$)')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E3_taux, axis=(1)) - np.mean(clim_E2_taux, axis=(1)), levels=np.linspace(-0.2,0.2,21), extend='both', cmap='PuOr_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('Zonal wind stress [N/m²]')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Monthly mean TAUX (5S-5N) (PI$^{off}_{45}$ - PI$^{on}_{45}$)')


#%%
fig, axs = plt.subplots(1, 1, figsize=(12, 4))
cf = axs.contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E4, axis=(1)), levels=np.linspace(22,30,21), extend='both', cmap='Spectral_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs.set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs.set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs)
colorbar.set_label('SST [°C]')
axs.set_ylabel('Month')
axs.set_title('b) Climatological monthly mean SST (5S-5N) for PI$^{off}_{18}$')

fig, axs = plt.subplots(1, 1, figsize=(8, 4))
cf2 = axs.contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E2, axis=(1)), levels=np.linspace(22,30,21), extend='both', cmap='Spectral_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs.set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs.set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf2, ax=axs)
colorbar.set_label('Zonal wind stress [N/m²]')
axs.set_ylabel('Month')
axs.set_title('a) Climatological monthly mean SST (5S-5N) for PI$^{on}_{45}$')

fig, axs = plt.subplots(1, 1, figsize=(8, 4))
cf = axs.contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E3, axis=(1)), levels=np.linspace(22,30,21), extend='both', cmap='Spectral_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs.set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs.set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs)
colorbar.set_label('SST (°C)')
axs.set_ylabel('Month')
axs.set_title('a) Climatological monthly mean SST (5S-5N) for PI$^{off}_{45}$')


# %% Nino indices

def TrendRemover(time, data, trend_type):
	"""Removes trend of choice"""
	
	rank = polyfit(time, data, trend_type)
	fitting = 0.0 
		
	for rank_i in range(len(rank)):
			
		fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))

	data -= fitting
	
	return data

def MonthRemover(time, data_all):
    """Removes the monthly average for each time series"""
    
    for month_i in range(12):
        month_index             = np.arange(month_i, len(time), 12)
        data_all[month_index]   = data_all[month_index] - np.mean(data_all[month_index], axis = 0)

    return data_all

def Nino_timeseries_numpy(data, latitude, longitude, time, mode='Nino34', trend=None, remove_month=None):
    """
    Calculate the Nino3 / 4 / 3.4 / 1+2 index time series from SST data using NumPy.
    
    Parameters:
        data: 3D NumPy array (time, latitude, longitude) of SST data.
        latitude: 1D NumPy array of latitude values.
        longitude: 1D NumPy array of longitude values.
        time: 1D NumPy array of time values.
        mode: str, the Nino region ('Nino3', 'Nino4', 'Nino34', or 'Nino12').
        trend: str, the type of trend to remove ('linear' or None).
    
    Returns:
        Nino: 1D NumPy array of the Nino index time series.
    """
    # Latitude and longitude bounds
    lat_min, lat_max = -5, 5
    if mode == 'Nino4':
        minlon, maxlon = 160, 210
    elif mode == 'Nino34':
        minlon, maxlon = 190, 240
    elif mode == 'Nino3':
        minlon, maxlon = 210, 270
    elif mode == 'Nino12':
        minlat, maxlat = -10, 0
        minlon, maxlon = 270, 280
    else:
        raise ValueError("Invalid mode. Choose from 'Nino3', 'Nino4', 'Nino34', or 'Nino12'.")

    # Subset the data based on latitude and longitude bounds (lat is more or less rectangular in the equator region)
    lat_min_index	= (fabs(latitude[:,0] - lat_min)).argmin()
    lat_max_index	= (fabs(latitude[:,0] - lat_max)).argmin() + 1
    lon_min_index	= (fabs(longitude[0,:] - minlon)).argmin()
    lon_max_index	= (fabs(longitude[0,:] - maxlon)).argmin() + 1
    
    print(lat_min_index)
    print(lat_max_index)
    print(lon_min_index)
    print(lon_max_index)
    
    # Apply the masks to the data
    data_subset = data[:, lat_min_index:lat_max_index, lon_min_index:lon_max_index]
    lat_subset = latitude[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
    lon_subset = longitude[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
    
    print(lat_subset[:,0])
    print(lon_subset[0,:])
    
    print(data_subset.shape)
    print(lat_subset.shape)
    print(lon_subset.shape)
    
    plt.figure()
    plt.contourf(lon_subset, lat_subset, data_subset[0, :, :])

    # Print the latitude and longitude values used in the mean calculation
    print("Latitudes used in the mean calculation:")
    print(lat_subset)
    print("Longitudes used in the mean calculation:")
    print(lon_subset)
    
    # Pre-processing
    
    if remove_month == 1:
        print('Monthly signal is removed\n')
        data_subset = MonthRemover(time, data_subset)
        
    if trend > 0:
        print('Trend is removed\n')
        for lat_i in range(len(lat_subset)):
            for lon_i in range(len(lon_subset[0])):
                data_subset[:, lat_i, lon_i] = TrendRemover(time, data_subset[:, lat_i, lon_i], trend)
                
    plt.figure()
    plt.contourf(data_subset.mean(axis=0))

    # Calculate the mean over latitude and longitude
    Nino = np.nanmean(data_subset, axis=(1, 2))  # Mean over lat and lon dimensions

    # Remove trend if specified
    #if trend == "linear":
    #    t = np.arange(len(Nino))  # Time indices
    #    p = np.polyfit(t, Nino, 1)  # Linear trend coefficients
    #    trend_line = np.polyval(p, t)  # Evaluate the trend line
    #    Nino = Nino - trend_line  # Remove the trend

    return Nino

Nino_34_E1 = Nino_timeseries_numpy(SST_month_E1, lat, lon, time_month_E1, mode='Nino34', trend=1, remove_month=1)
Nino_34_E2 = Nino_timeseries_numpy(SST_month_E2, lat, lon, time_month_E2, mode='Nino34', trend=1, remove_month=1)
Nino_34_E3 = Nino_timeseries_numpy(SST_month_E3, lat, lon, time_month_E3, mode='Nino34', trend=1, remove_month=1)
Nino_34_E4 = Nino_timeseries_numpy(SST_month_E4, lat, lon, time_month_E4, mode='Nino34', trend=1, remove_month=1)
# %% Plot Nino time series for the 4 branches

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

plt.suptitle('Nino 3.4 index')

axs[0,0].plot(time_month_E1, Nino_34_E1, 'k')
axs[0,0].axhline(0.5, color='red')
axs[0,0].axhline(-0.5, color='blue')
axs[0,0].set_title('a) AMOC on ($F_H$ = 0.18Sv)')
axs[0,0].grid()

axs[0,1].plot(time_month_E2, Nino_34_E2, 'k')
axs[0,1].axhline(0.5, color='red')
axs[0,1].axhline(-0.5, color='blue')
axs[0,1].set_title('a) AMOC on ($F_H$ = 0.45Sv)')
axs[0,1].grid()

axs[1,0].plot(time_month_E3, Nino_34_E3, 'k')
axs[1,0].axhline(0.5, color='red')
axs[1,0].axhline(-0.5, color='blue')
axs[1,0].set_title('a) AMOC off ($F_H$ = 0.18Sv)')
axs[1,0].grid()

axs[1,1].plot(time_month_E4, Nino_34_E4, 'k')
axs[1,1].axhline(0.5, color='red')
axs[1,1].axhline(-0.5, color='blue')
axs[1,1].set_title('a) AMOC off ($F_H$ = 0.45Sv)')
axs[1,1].grid()

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'Nino34_index_branches.pdf')
plt.show()
# %%

print('Standard deviations:')
print('E1:', np.std(Nino_34_E1))
print('E2:', np.std(Nino_34_E2))
print('E3:', np.std(Nino_34_E3))
print('E4:', np.std(Nino_34_E4))

print('Variances:')
print('E1:', np.var(Nino_34_E1))
print('E2:', np.var(Nino_34_E2))
print('E3:', np.var(Nino_34_E3))
print('E4:', np.var(Nino_34_E4))

print('Mean:')
print('E1:', np.mean(Nino_34_E1))
print('E2:', np.mean(Nino_34_E2))
print('E3:', np.mean(Nino_34_E3))
print('E4:', np.mean(Nino_34_E4))

# %%

el_nino_E1 = np.sum(Nino_34_E1 > 0.5)
la_nina_E1 = np.sum(Nino_34_E1 < -0.5)

el_nino_E2 = np.sum(Nino_34_E2 > 0.5)
la_nina_E2 = np.sum(Nino_34_E2 < -0.5)

el_nino_E3 = np.sum(Nino_34_E3 > 0.5)
la_nina_E3 = np.sum(Nino_34_E3 < -0.5)

el_nino_E4 = np.sum(Nino_34_E4 > 0.5)
la_nina_E4 = np.sum(Nino_34_E4 < -0.5)

print('E1 - El Niño events:', el_nino_E1, 'La Niña events:', la_nina_E1)
print('E2 - El Niño events:', el_nino_E2, 'La Niña events:', la_nina_E2)
print('E3 - El Niño events:', el_nino_E3, 'La Niña events:', la_nina_E3)
print('E4 - El Niño events:', el_nino_E4, 'La Niña events:', la_nina_E4)
# %% Multitaper spectrum of the Nino 3.4 index for the 4 branches


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
fs = 12.0 #monthly data, so sampling frequency is 12 per year
NW = 3
Kmax = 5
nsurr = 2000
ci_level = 95

# --- compute absolute PSDs + envelopes ---
f_nino_E1,  S_nino_E1,  ci_nino_E1,  phi_nino_E1  = mtm_psd_ar1_ci_abs(Nino_34_E1,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=0)
f_nino_E2,  S_nino_E2,  ci_nino_E2,  phi_nino_E2  = mtm_psd_ar1_ci_abs(Nino_34_E2,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=0)
f_nino_E3,  S_nino_E3,  ci_nino_E3,  phi_nino_E3  = mtm_psd_ar1_ci_abs(Nino_34_E3,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=0)
f_nino_E4,  S_nino_E4,  ci_nino_E4,  phi_nino_E4  = mtm_psd_ar1_ci_abs(Nino_34_E4,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,), seed=0)

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_nino_E1,  Sp_nino_E1,  CIp_nino_E1  = to_period_sorted(f_nino_E1,  S_nino_E1,  ci_nino_E1[ci_level])
per_nino_E2,  Sp_nino_E2,  CIp_nino_E2  = to_period_sorted(f_nino_E2,  S_nino_E2,  ci_nino_E2[ci_level])
per_nino_E3,  Sp_nino_E3,  CIp_nino_E3  = to_period_sorted(f_nino_E3,  S_nino_E3,  ci_nino_E3[ci_level])
per_nino_E4,  Sp_nino_E4,  CIp_nino_E4  = to_period_sorted(f_nino_E4,  S_nino_E4,  ci_nino_E4[ci_level])
# %%

period_xlim = (1, 12)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

spectra_panels = [
    (axes[0], per_nino_E1, Sp_nino_E1, CIp_nino_E1, r"a) PI$_{18}$ - Nino 3.4 index", "royalblue", "MT spectrum PI$^{\mathrm{on}}_{18}$"),
    (axes[1], per_nino_E2, Sp_nino_E2, CIp_nino_E2, r"b) PI$_{45}$ - Nino 3.4 index", "royalblue", "MT spectrum PI$^{\mathrm{on}}_{45}$"),
    (axes[1], per_nino_E3, Sp_nino_E3, CIp_nino_E3, r"a) PI$_{45}$ - Nino 3.4 index", "indianred", "MT spectrum PI$^{\mathrm{off}}_{45}$"),
    (axes[0], per_nino_E4, Sp_nino_E4, CIp_nino_E4, r"b) PI$_{18}$ - Nino 3.4 index", "indianred", "MT spectrum PI$^{\mathrm{off}}_{18}$")
]

for ax, per, Sp, CIp, title, col, label in spectra_panels:
    ax.plot(per, Sp, lw=1.8, color=col, label=label)
    ax.plot(per, CIp, lw=1.2, color="black", ls="--")
    ax.set_ylim(0, 0.021)
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

axes[0].set_ylabel(r"Power [$^\circ$C$^2$ yr$^{-1}$]", fontsize=11)
axes[0].set_xlabel("Period [model years]", fontsize=11)
axes[1].set_xlabel("Period [model years]", fontsize=11)

axes[0].legend(frameon=False, loc="upper left")
axes[1].legend(frameon=False, loc="upper left")  

fig.tight_layout()
plt.show()
# %% Make a combined Figure including the SST standard deviation maps, SST and zonal wind stress climatological monthly mean maps, Nino 3.4 index time series and spectra for the 4 branches. This will be a comprehensive figure summarizing the main differences between the branches in terms of mean state, variability and ENSO characteristics.

fig = plt.figure(figsize=(14, 12))

ax00 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax01 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
ax10 = fig.add_subplot(3, 2, 3)
ax11 = fig.add_subplot(3, 2, 4)
ax20 = fig.add_subplot(3, 2, 5)
ax21 = fig.add_subplot(3, 2, 6)

# --------------------------------------------------
# Helpful longitude handling
# --------------------------------------------------
lon_vals = lon.mean(axis=0) if getattr(lon, "ndim", 1) == 2 else lon
lat_vals = lat[:, 0] if getattr(lat, "ndim", 1) == 2 else lat

# Niño3.4 box
if np.nanmax(lon_vals) > 180:
    lon_box_min, lon_box_max = 190, 240   # 170W–120W on 0–360 grid
else:
    lon_box_min, lon_box_max = -170, -120
lat_box_min, lat_box_max = -5, 5

# --------------------------------------------------
# 1) MAP PANELS
# --------------------------------------------------
levels_map = np.linspace(-0.6, 0.6, 21)

map1 = np.std(SST_month_E4_anom_DT, axis=0) - np.std(SST_month_E1_anom_DT, axis=0)
map2 = np.std(SST_month_E3_anom_DT, axis=0) - np.std(SST_month_E2_anom_DT, axis=0)

c1 = ax00.contourf(
    lon, lat, map1,
    transform=ccrs.PlateCarree(),
    levels=levels_map, cmap="RdBu_r", extend="both")
cb1 = fig.colorbar(c1, ax=ax00, orientation="horizontal", pad=0.08)
cb1.set_label("std SST [°C]")

ax00.set_title(r"a) std SST (annual, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")
ax00.set_xlim(-60, 110)
ax00.set_ylim(-20, 20)
ax00.coastlines()

#xticks = [160, 180, 200, 220, 240, 260, 280]
#xticklabels = ['160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W']

#ax00.set_xticks(np.arange(-30, 111, 30), crs=ccrs.PlateCarree())
ax00.set_yticks(np.arange(-20, 21, 10), crs=ccrs.PlateCarree())
ax00.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax00.yaxis.set_major_formatter(cticker.LatitudeFormatter())

rect1 = Rectangle(
    (lon_box_min, lat_box_min),
    lon_box_max - lon_box_min,
    lat_box_max - lat_box_min,
    linewidth=2, edgecolor="k", facecolor="none",
    transform=ccrs.PlateCarree())
ax00.add_patch(rect1)

c2 = ax01.contourf(
    lon, lat, map2,
    transform=ccrs.PlateCarree(),
    levels=levels_map, cmap="RdBu_r", extend="both")
cb2 = fig.colorbar(c2, ax=ax01, orientation="horizontal", pad=0.08)
cb2.set_label("std SST [°C]")

ax01.set_title(r"b) std SST (annual, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)")
ax01.set_xlim(-60, 110)
ax01.set_ylim(-20, 20)
ax01.coastlines()

ax01.set_xticks(np.arange(160, -80, 30), crs=ccrs.PlateCarree())
ax01.set_yticks(np.arange(-20, 21, 20), crs=ccrs.PlateCarree())
ax01.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax01.yaxis.set_major_formatter(cticker.LatitudeFormatter())

rect2 = Rectangle(
    (lon_box_min, lat_box_min),
    lon_box_max - lon_box_min,
    lat_box_max - lat_box_min,
    linewidth=2, edgecolor="k", facecolor="none",
    transform=ccrs.PlateCarree())
ax01.add_patch(rect2)

# --------------------------------------------------
# 2) MONTHLY MEAN EQUATORIAL TRANSECTS
# Assuming clim_E* has shape (12, lat, lon)
# and you want the tropical mean over latitude, e.g. 5S–5N
# --------------------------------------------------
months = np.arange(1, 13)

# If clim_E* is already averaged over 5S–5N, keep your original mean(axis=1) or adapt as needed.
hov1 = np.mean(clim_E4, axis=1) - np.mean(clim_E1, axis=1)
hov2 = np.mean(clim_E3, axis=1) - np.mean(clim_E2, axis=1)

levels_hov = np.linspace(-1, 1, 21)

cf1 = ax10.contourf(
    lon_vals, months, hov1,
    levels=levels_hov, cmap="RdBu_r", extend="both")
cb3 = fig.colorbar(cf1, ax=ax10)
cb3.set_label("SST [°C]")

levels_hov_clim = np.arange(22, 30, 1)

cs = ax10.contour(
    lon_vals,
    months,
    np.mean(clim_E1, axis=1),
    levels=levels_hov_clim,
    colors="k",
    linewidths=0.5)

ax10.clabel(cs, inline=True, fontsize=8)
ax10.set_xlim(160, 280)
ax10.set_ylim(1, 12)
ax10.set_ylabel("Month")
ax10.set_title(r"c) Monthly mean SST (5°S–5°N) (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)")

xticks = [160, 180, 200, 220, 240, 260, 280]
xticklabels = ['160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W']
ax10.set_xticks(xticks)
ax10.set_xticklabels(xticklabels)

cf2 = ax11.contourf(
    lon_vals, months, hov2,
    levels=levels_hov, cmap="RdBu_r", extend="both")
cb4 = fig.colorbar(cf2, ax=ax11)
cb4.set_label("SST [°C]")

cs = ax11.contour(
    lon_vals,
    months,
    np.mean(clim_E2, axis=1),
    levels=levels_hov_clim,
    colors="k",
    linewidths=0.5)

ax10.clabel(cs, inline=True, fontsize=8)

ax11.set_xlim(160, 280)
ax11.set_ylabel("Month")
ax11.set_title(r"d) Monthly mean SST (5°S–5°N) (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)")
ax11.set_xticks(xticks)
ax11.set_xticklabels(xticklabels)

# --------------------------------------------------
# 3) SPECTRA PANELS
# --------------------------------------------------
spectra_info = [
    (ax20,
     per_nino_E1, Sp_nino_E1, CIp_nino_E1, "royalblue", r"MT spectrum PI$^{\mathrm{on}}_{18}$",
     per_nino_E4, Sp_nino_E4, CIp_nino_E4, "indianred", r"MT spectrum PI$^{\mathrm{off}}_{18}$",
     r"e) PI$_{18}$ - Niño 3.4 index"),
    (ax21,
     per_nino_E2, Sp_nino_E2, CIp_nino_E2, "royalblue", r"MT spectrum PI$^{\mathrm{on}}_{45}$",
     per_nino_E3, Sp_nino_E3, CIp_nino_E3, "indianred", r"MT spectrum PI$^{\mathrm{off}}_{45}$",
     r"f) PI$_{45}$ - Niño 3.4 index")]

for ax, per1, sp1, cip1, col1, lab1, per2, sp2, cip2, col2, lab2, title in spectra_info:
    ax.plot(per1, sp1, lw=1.8, color=col1, label=lab1)
    ax.plot(per1, cip1, lw=1.2, color=col1, ls="--", alpha=0.8)

    ax.plot(per2, sp2, lw=1.8, color=col2, label=lab2)
    ax.plot(per2, cip2, lw=1.2, color=col2, ls="--", alpha=0.8)

    ax.set_ylim(0, 0.01)
    ax.set_xlim(1, *period_xlim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Period [model years]")
    ax.set_ylabel("Power [K² / yr⁻¹]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(directory_figures + 'Figure_std_ssts_monthlycycle_ninoindex.pdf')
plt.show()

# %% 

fig = plt.figure(figsize=(14, 12))

ax00 = fig.add_subplot(3, 2, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax01 = fig.add_subplot(3, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
ax20 = fig.add_subplot(3, 2, 3)
ax21 = fig.add_subplot(3, 2, 4)

# --------------------------------------------------
# Helpful longitude handling
# --------------------------------------------------
lon_vals = lon.mean(axis=0) if getattr(lon, "ndim", 1) == 2 else lon
lat_vals = lat[:, 0] if getattr(lat, "ndim", 1) == 2 else lat

# Niño3.4 box
if np.nanmax(lon_vals) > 180:
    lon_box_min, lon_box_max = 190, 240   # 170W–120W on 0–360 grid
else:
    lon_box_min, lon_box_max = -170, -120
lat_box_min, lat_box_max = -5, 5

# --------------------------------------------------
# 1) MAP PANELS
# --------------------------------------------------
levels_map = np.linspace(-0.6, 0.6, 21)

map1 = np.std(SST_month_E4_anom_DT, axis=0) - np.std(SST_month_E1_anom_DT, axis=0)
map2 = np.std(SST_month_E3_anom_DT, axis=0) - np.std(SST_month_E2_anom_DT, axis=0)

c1 = ax00.contourf(
    lon, lat, map1,
    transform=ccrs.PlateCarree(),
    levels=levels_map, cmap="RdBu_r", extend="both")
cb1 = fig.colorbar(c1, ax=ax00, orientation="horizontal", pad=0.08)
cb1.set_label("Standard deviation difference [°C]")

ax00.set_title(r"a) Standard deviation SST (monthly, PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)", fontsize=17)
ax00.set_xlim(-60, 110)
ax00.set_ylim(-20, 20)
ax00.coastlines()

#xticks = [160, 180, 200, 220, 240, 260, 280]
#xticklabels = ['160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W']

#ax00.set_xticks(np.arange(-30, 111, 30), crs=ccrs.PlateCarree())
ax00.set_yticks(np.arange(-20, 21, 10), crs=ccrs.PlateCarree())
ax00.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax00.yaxis.set_major_formatter(cticker.LatitudeFormatter())

rect1 = Rectangle(
    (lon_box_min, lat_box_min),
    lon_box_max - lon_box_min,
    lat_box_max - lat_box_min,
    linewidth=2, edgecolor="k", facecolor="none",
    transform=ccrs.PlateCarree())
ax00.add_patch(rect1)

c2 = ax01.contourf(
    lon, lat, map2,
    transform=ccrs.PlateCarree(),
    levels=levels_map, cmap="RdBu_r", extend="both")
cb2 = fig.colorbar(c2, ax=ax01, orientation="horizontal", pad=0.08)
cb2.set_label("Standard deviation difference [°C]")

ax01.set_title(r"b) Standard deviation SST (monthly, PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)", fontsize=17)
ax01.set_xlim(-60, 110)
ax01.set_ylim(-20, 20)
ax01.coastlines()

ax01.set_xticks(np.arange(160, -80, 30), crs=ccrs.PlateCarree())
ax01.set_yticks(np.arange(-20, 21, 20), crs=ccrs.PlateCarree())
ax01.xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax01.yaxis.set_major_formatter(cticker.LatitudeFormatter())

rect2 = Rectangle(
    (lon_box_min, lat_box_min),
    lon_box_max - lon_box_min,
    lat_box_max - lat_box_min,
    linewidth=2, edgecolor="k", facecolor="none",
    transform=ccrs.PlateCarree())
ax01.add_patch(rect2)

# --------------------------------------------------
# 3) SPECTRA PANELS
# --------------------------------------------------
spectra_info = [
    (ax20,
     per_nino_E1, Sp_nino_E1, CIp_nino_E1, "royalblue", r"MT spectrum PI$^{\mathrm{on}}_{18}$",
     per_nino_E4, Sp_nino_E4, CIp_nino_E4, "indianred", r"MT spectrum PI$^{\mathrm{off}}_{18}$",
     r"c) PI$_{18}$ - Niño 3.4 index"),
    (ax21,
     per_nino_E2, Sp_nino_E2, CIp_nino_E2, "royalblue", r"MT spectrum PI$^{\mathrm{on}}_{45}$",
     per_nino_E3, Sp_nino_E3, CIp_nino_E3, "indianred", r"MT spectrum PI$^{\mathrm{off}}_{45}$",
     r"d) PI$_{45}$ - Niño 3.4 index")]
for ax, per1, sp1, cip1, col1, lab1, per2, sp2, cip2, col2, lab2, title in spectra_info:
    ax.plot(per1, sp1, lw=1.8, color=col1, label=lab1)
    ax.plot(per1, cip1, lw=1.2, color=col1, ls="--", alpha=0.8)

    ax.plot(per2, sp2, lw=1.8, color=col2, label=lab2)
    ax.plot(per2, cip2, lw=1.2, color=col2, ls="--", alpha=0.8)

    ax.set_ylim(0, 0.01)
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("Period [model years]", fontsize=14)
    ax.set_ylabel("Power [°C² yr⁻¹]", fontsize=14)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=15)

plt.tight_layout()
plt.savefig(directory_figures + 'Figure_std_ssts_ninoindex.pdf')
plt.show()

# %%
