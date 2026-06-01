#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 13:48:16 2025

@author: 6008399

2m air temperature of equilibrium branches

"""


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
import ruptures as rpt
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
import statsmodels.api as sm
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.colors as mcolors
import cartopy.mpl.ticker as cticker

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

def Welch(data_1, data_2):
    """Conducts the Welch t-test"""
	
    #Determine the means
    mean_1	= np.mean(data_1)
    mean_2	= np.mean(data_2)
	
    #Determine the corrected sample standard deviations
    std_1	= np.sqrt(1.0 / (len(data_1) - 1) * np.sum((data_1 - mean_1)**2.0))
    std_2	= np.sqrt(1.0 / (len(data_2) - 1) * np.sum((data_2 - mean_2)**2.0))

    #Determine the Welch t-value
    t_welch	= (mean_1 - mean_2) / np.sqrt((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2)))

    #Determine the degrees of freedome (dof)
    dof	= ((std_1**2.0 / len(data_1)) + (std_2**2.0 / len(data_2)))**2.0 / ((std_1**4.0 / (len(data_1)**2.0 * (len(data_1) - 1))) + (std_2**4.0 / (len(data_2)**2.0 * (len(data_2) - 1))))

    #Get the significance levels and the corresponding critical values (two-sided)
    sig_levels 	= np.arange(50, 100, 0.5) / 100.0
    t_crit		= stats.t.ppf((1.0 + sig_levels) / 2.0, dof)

    #Get the indices where the significance is exceeding the critical values
    sig_index	= np.where(fabs(t_welch) > t_crit)[0]
    significant	= 0.0

    if len(sig_index) > 0:
        #If there are significance values, take the highest significant level
        significant = sig_levels[sig_index[-1]]

    return significant

#%%

region      = 'global'
    
#Read in data
fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach600_year_999-1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_1_annual   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_12-14_brach600_year_999-1100.nc', 'r')

TEMP_1_DJF   = fh.variables['TS'][:]  #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_6-8_brach600_year_999-1100.nc', 'r')

TEMP_1_JJA   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach1500_year_1899-2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_2_annual   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_12-14_brach1500_year_1899-2000.nc', 'r')

TEMP_2_DJF   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_6-8_brach1500_year_1899-2000.nc', 'r')

TEMP_2_JJA   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach2900_year_2900-3500.nc', 'r')

time3           = fh.variables['time'][400:500] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_3_annual   = fh.variables['TS'][400:500,:,:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_12-14_brach2900_year_2900-3500.nc', 'r')

TEMP_3_DJF   = fh.variables['TS'][400:500,:,:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_6-8_brach2900_year_2900-3500.nc', 'r')

TEMP_3_JJA   = fh.variables['TS'][400:500,:,:]   #Mean reference air temperature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach3800_year_4199-4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_4_annual   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_12-14_brach3800_year_4199-4300.nc', 'r')

TEMP_4_DJF   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_6-8_brach3800_year_4199-4300.nc', 'r')

TEMP_4_JJA   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs:
    ax.coastlines()

contourf1 = axs[0].contourf(
    lon, lat, np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0), transform=ccrs.PlateCarree(),
    levels=np.linspace(-10, 10, 41), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
cbar1.set_label('Temperature difference [$^\circ$C]', fontsize=12)
cbar1.set_ticks([-9, -6, -3, 0, 3, 6, 9])
axs[0].set_xticks(np.arange(-180,181, 60), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0].xaxis.set_major_formatter(lon_formatter)
axs[0].set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0].yaxis.set_major_formatter(lat_formatter)
#axs[0].set_ylim(1000, 0)
#axs[0].set_ylabel('Latitude [$^\circ$N]', fontsize=12)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0].set_title('a) Annual 2m temperature ($F_H$ = 0.18Sv)', fontsize=14)

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        #Determine significant difference
        p_value = Welch(TEMP_1_annual[:, lat_i, lon_i], TEMP_4_annual[:, lat_i, lon_i])

        if p_value <= 0.95:
            #Non-significant difference
            axs[0].scatter(lon[lon_i], lat[lat_i], marker = 'o', edgecolor = 'k' , s = 6, facecolors='none')


contourf2 = axs[1].contourf(
    lon, lat, np.mean(TEMP_3_annual, axis=0) - np.mean(TEMP_2_annual, axis=0),
    levels=np.linspace(-10, 10, 41), cmap='RdBu_r', extend='both'
)
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('Temperature difference [$^\circ$C]', fontsize=12)
cbar2.set_ticks([-9, -6, -3, 0, 3, 6, 9])
axs[1].set_xticks(np.arange(-180,181, 60), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1].xaxis.set_major_formatter(lon_formatter)
axs[1].set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1].yaxis.set_major_formatter(lat_formatter)
axs[0].set_ylim(-10, 10)
#axs[1].set_ylabel('Latitude [$^\circ$N]', fontsize=12)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1].set_title('b) Annual 2m temperature ($F_H$ = 0.45Sv)', fontsize=14)

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        #Determine significant difference
        p_value = Welch(TEMP_2_annual[:, lat_i, lon_i], TEMP_3_annual[:, lat_i, lon_i])

        if p_value <= 0.95:
            #Non-significant difference
            axs[1].scatter(lon[lon_i], lat[lat_i], marker = 'o', edgecolor = 'k' , s = 6, facecolors='none')

plt.tight_layout()
#plt.savefig(directory_figures + 'time_mean_TEMP_difference_'+str(region)+'_OFF_ON_forcing_018_045Sv.pdf')
plt.show()

#%%

fig, ax = plt.subplots(1,1, figsize = (10,4), subplot_kw={'projection': ccrs.PlateCarree()})

ax.coastlines()

contourf1 = ax.contourf(
    lon, lat, np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0), transform=ccrs.PlateCarree(),
    levels=np.linspace(-10, 10, 41), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=ax)
cbar1.set_label('Temperature difference [$^\circ$C]', fontsize=12)
cbar1.set_ticks([-9, -6, -3, 0, 3, 6, 9])
ax.set_xticks(np.arange(-180,181, 60), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)
#axs[0].set_ylim(1000, 0)
#axs[0].set_ylabel('Latitude [$^\circ$N]', fontsize=12)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
ax.set_title('Annual 2m temperature ($\overline{F_H}$ = 0.18Sv)', fontsize=14)

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        #Determine significant difference
        p_value = Welch(TEMP_1_annual[:, lat_i, lon_i], TEMP_4_annual[:, lat_i, lon_i])

        if p_value <= 0.95:
            #Non-significant difference
            ax.scatter(lon[lon_i], lat[lat_i], marker = 'o', edgecolor = 'k' , s = 6, facecolors='none')
            
plt.tight_layout()
plt.savefig(directory_figures + 'time_mean_TEMP_difference_'+str(region)+'_OFF_ON_forcing_018Sv_annual.pdf')
plt.show()

#%%

fig, ax = plt.subplots(1,1, figsize = (10,4), subplot_kw={'projection': ccrs.PlateCarree()})

ax.coastlines()

contourf1 = ax.contourf(
    lon, lat, (np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0)) - (np.mean(TEMP_3_annual, axis=0) - np.mean(TEMP_2_annual, axis=0)), transform=ccrs.PlateCarree(),
    levels=np.linspace(-3, 3, 41), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=ax)
cbar1.set_label('Temperature difference [$^\circ$C]', fontsize=12)
#cbar1.set_ticks([-9, -6, -3, 0, 3, 6, 9])
ax.set_xticks(np.arange(-180,181, 60), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)
#axs[0].set_ylim(1000, 0)
#axs[0].set_ylabel('Latitude [$^\circ$N]', fontsize=12)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
ax.set_title('Annual 2m temperature (Difference PI$_{18}$ minus PI$_{45}$)', fontsize=14)

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        #Determine significant difference
        p_value = Welch(TEMP_1_annual[:, lat_i, lon_i], TEMP_4_annual[:, lat_i, lon_i])

        if p_value <= 0.95:
            #Non-significant difference
            ax.scatter(lon[lon_i], lat[lat_i], marker = 'o', edgecolor = 'k' , s = 6, facecolors='none')
            
plt.tight_layout()
#plt.savefig(directory_figures + 'time_mean_TEMP_difference_'+str(region)+'_OFF_ON_forcing_018Sv_annual.pdf')
plt.show()

fig, ax = plt.subplots(1,1, figsize = (10,4), subplot_kw={'projection': ccrs.PlateCarree()})

ax.coastlines()

contourf1 = ax.contourf(
    lon, lat, (np.mean(TEMP_4_DJF, axis=0) - np.mean(TEMP_1_DJF, axis=0)) - (np.mean(TEMP_3_DJF, axis=0) - np.mean(TEMP_2_DJF, axis=0)), transform=ccrs.PlateCarree(),
    levels=np.linspace(-3, 3, 41), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=ax)
cbar1.set_label('Temperature difference [$^\circ$C]', fontsize=12)
#cbar1.set_ticks([-9, -6, -3, 0, 3, 6, 9])
ax.set_xticks(np.arange(-180,181, 60), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter)
#axs[0].set_ylim(1000, 0)
#axs[0].set_ylabel('Latitude [$^\circ$N]', fontsize=12)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
ax.set_title('DJF 2m temperature (Difference $\overline{F_H}$ = 0.18Sv minus $\overline{F_H}$ = 0.45Sv)', fontsize=14)

for lat_i in range(0, len(lat), 3):
    for lon_i in range(0, len(lon), 3):
        #Determine significant difference
        p_value = Welch(TEMP_1_annual[:, lat_i, lon_i], TEMP_4_annual[:, lat_i, lon_i])

        if p_value <= 0.95:
            #Non-significant difference
            ax.scatter(lon[lon_i], lat[lat_i], marker = 'o', edgecolor = 'k' , s = 6, facecolors='none')
            
plt.tight_layout()
#plt.savefig(directory_figures + 'time_mean_TEMP_difference_'+str(region)+'_OFF_ON_forcing_018Sv_annual.pdf')
plt.show()



#%%

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

# Create a 3x2 subplot
fig, axs = plt.subplots(3, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# Define the data for each subplot
data = {
    'a) Annual ($F_H$ = 0.18Sv)': (TEMP_1_annual, TEMP_4_annual),
    'b) Annual ($F_H$ = 0.45Sv)': (TEMP_2_annual, TEMP_3_annual),
    'c) DJF ($F_H$ = 0.18Sv)': (TEMP_1_DJF, TEMP_4_DJF),
    'd) DJF ($F_H$ = 0.45Sv)': (TEMP_2_DJF, TEMP_3_DJF),
    'e) JJA ($F_H$ = 0.18Sv)': (TEMP_1_JJA, TEMP_4_JJA),
    'f) JJA ($F_H$ = 0.45Sv)': (TEMP_2_JJA, TEMP_3_JJA),
}

# Iterate over the subplots and plot the data
for ax, (title, (TEMP_A, TEMP_B)) in zip(axs.flat, data.items()):
    # Add coastlines
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Contourf plot for the difference
    contourf = ax.contourf(
        lon, lat, np.mean(TEMP_B, axis=0) - np.mean(TEMP_A, axis=0),
        levels=np.linspace(-10, 10, 41), cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree()
    )
    cbar = fig.colorbar(contourf, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Temperature difference [$^\circ$C]', fontsize=12)
    cbar.set_ticks([-9, -6, -3, 0, 3, 6, 9])

    # Set gridlines and labels
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    # Add title
    ax.set_title(title, fontsize=14)

    # Add significance markers
    for lat_i in range(0, len(lat), 3):
        for lon_i in range(0, len(lon), 3):
            # Determine significant difference
            p_value = Welch(TEMP_A[:, lat_i, lon_i], TEMP_B[:, lat_i, lon_i])
            if p_value <= 0.95:
                ax.scatter(lon[lon_i], lat[lat_i], marker='o', edgecolor='k', s=6, facecolors='none', transform=ccrs.PlateCarree())

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig(directory_figures + 'time_mean_2m_TEMP_difference_annual_DJF_JJA_018_045Sv.pdf')
plt.show()