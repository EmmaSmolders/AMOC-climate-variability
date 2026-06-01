#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 09:17:41 2025

@author: 6008399

EOF AMV plotting

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



#lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory + 'EOF_AMV_SST_forward_month_1_12_moving_average_0_CESM_QE_year_600_1500.nc')
#lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory + 'EOF_AMV_SST_backward_month_1_12_moving_average_0_CESM_QE_year_2900_3800.nc')

lon_sst, lat_sst, eof_E1, time_E1, PC_E1_SST, VAR_E1_SST, EOF_E1_SST		= ReadinData(directory + 'EOF_AMV_SST_forward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_600_1500.nc')
lon_sst, lat_sst, eof_E2, time_E2, PC_E2_SST, VAR_E2_SST, EOF_E2_SST		= ReadinData(directory + 'EOF_AMV_SST_backward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_2900_3800.nc')

lon_temp, lat_temp, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory + 'EOF_AMV_TEMP_forward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_600_1500.nc')
lon, lat, eof_E1, time_E1, PC_E1_detrend1, VAR_E1_detrend1, EOF_E1_detrend1		= ReadinData(directory + 'EOF_AMV_TEMP_forward_month_1_12_lowpass_runmean_120mo_detrend1_CESM_QE_year_600_1500.nc')
lon, lat, eof_E1, time_E1_none, PC_E1_detrend1_none, VAR_E1_detrend1_none, EOF_E1_detrend1_none		= ReadinData(directory + 'EOF_AMV_TEMP_forward_month_1_12_lowpass_none_detrend1_CESM_QE_year_600_1500.nc')
lon, lat, eof_E1, time_E1_none, PC_E1_detrend2_none, VAR_E1_detrend2_none, EOF_E1_detrend2_none		= ReadinData(directory + 'EOF_AMV_TEMP_forward_month_1_12_lowpass_none_detrend2_CESM_QE_year_600_1500.nc')

lon_temp, lat_temp, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory + 'EOF_AMV_TEMP_backward_month_1_12_lowpass_runmean_120mo_detrend2_CESM_QE_year_2900_3800.nc')
lon, lat, eof_E2, time_E2, PC_E2_detrend1, VAR_E2_detrend1, EOF_E2_detrend1		= ReadinData(directory + 'EOF_AMV_TEMP_backward_month_1_12_lowpass_runmean_120mo_detrend1_CESM_QE_year_2900_3800.nc')
lon, lat, eof_E2, time_E2_none, PC_E2_detrend1_none, VAR_E2_detrend1_none, EOF_E2_detrend1_none		= ReadinData(directory + 'EOF_AMV_TEMP_backward_month_1_12_lowpass_none_detrend1_CESM_QE_year_2900_3800.nc')
lon, lat, eof_E2, time_E2_none, PC_E2_detrend2_none, VAR_E2_detrend2_none, EOF_E2_detrend2_none		= ReadinData(directory + 'EOF_AMV_TEMP_backward_month_1_12_lowpass_none_detrend2_CESM_QE_year_2900_3800.nc')

#%% Take first EOF and multiply with PC standard deviation to get typical anomaly pattern in [degC]

EOF_TEMP_E1 = EOF_E1[0,:,:] * PC_E1[0,:].std()
EOF_TEMP_E2 = EOF_E2[0,:,:] * PC_E2[0,:].std()

EOF2_TEMP_E1 = EOF_E1[1,:,:] * PC_E1[1,:].std()
EOF2_TEMP_E2 = EOF_E2[1,:,:] * PC_E2[1,:].std()

EOF_SST_E1 = EOF_E1_SST[0,:,:] * PC_E1_SST[0,:].std()
EOF_SST_E2 = EOF_E2_SST[0,:,:] * PC_E2_SST[0,:].std()

EOF2_SST_E1 = EOF_E1_SST[1,:,:] * PC_E1_SST[1,:].std()
EOF2_SST_E2 = EOF_E2_SST[1,:,:] * PC_E2_SST[1,:].std()

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

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('Depth averaged (100-300m) temperatures', fontsize=15)
    
c1 = axs[0,0].contourf(lon_temp, lat_temp, EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.05,0.05,21), extend='both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) First EOF AMOC on (var.ex. = '+str(int(VAR_E1[0]*100))+'%)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[0,1].contourf(lon_temp, lat_temp, EOF_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) First EOF AMOC off (var.ex. = '+str(int(VAR_E2[0]*100))+'%)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1,0].contourf(lon_temp, lat_temp, EOF2_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) Second EOF AMOC on (var.ex. = '+str(int(VAR_E1[1]*100))+'%)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,71,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[1,1].contourf(lon_temp, lat_temp, -EOF2_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Second EOF AMOC off (var.ex. = '+str(int(VAR_E2[1]*100))+'%)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,71,20), crs=ccrs.PlateCarree())
axs[1,1].set_ylim(10,70)
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'EOF_TEMP_depthaveraged_ATLANTIC_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show()    

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('SSTs', fontsize=15)
    
c1 = axs[0,0].contourf(lon_sst, lat_sst, -EOF_SST_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend='both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) First EOF AMOC on (var.ex. = '+str(int(VAR_E1_SST[0]*100))+'%)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,71,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_ylim(10,70)
axs[0,0].set_xlim(-90,20)

c2 = axs[0,1].contourf(lon_sst, lat_sst, -EOF_SST_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) First EOF AMOC off (var.ex. = '+str(int(VAR_E2_SST[0]*100))+'%)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(10,71,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_ylim(10,70)
axs[0,1].set_xlim(-90,20)

c1 = axs[1,0].contourf(lon_sst, lat_sst, -EOF2_SST_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) Second EOF AMOC on (var.ex. = '+str(int(VAR_E1_SST[1]*100))+'%)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(10,71,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_ylim(10,70)
axs[1,0].set_xlim(-90,20)

c2 = axs[1,1].contourf(lon_sst, lat_sst, -EOF2_SST_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Second EOF AMOC off (var.ex. = '+str(int(VAR_E2_SST[1]*100))+'%)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(10,71,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_ylim(10,70)
axs[1,1].set_xlim(-90,20)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'EOF_SST_ATLANTIC_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show()   

#%%

def ReadinDataIFRAC(filename):

	fh = netcdf.Dataset(filename, 'r')

	lat	       = fh.variables['lat'][:,0:100]     		  #Latitudes (degrees N)
	lon	       = fh.variables['lon'][:,0:100]     	      #Longitudes(degrees E)
	time	   = fh.variables['time'][:]		              #Time (model year)
	time_month = fh.variables['time_month'][:]		          #Time (model year)
	IFRAC        = fh.variables['ice'][:,:,0:100]     	  #surface heat flux (W/m2)
	IFRAC_month  = fh.variables['ice_month'][:,:,0:100]    #Monthly surface heat flux (W/m2)

	fh.close()

	return lat, lon, time, time_month, IFRAC, IFRAC_month

lat_ice, lon_ice, time_forward, time_month_forward, IFRAC_forward, IFRAC_month_forward 	= ReadinDataIFRAC(directory_data+'/Atmosphere/IFRAC_Atlantic_year_600-1500_month_1-12_QE.nc')
lat_ice, lon_ice, time_backward, time_month_backward, IFRAC_backward, IFRAC_month_backward 	= ReadinDataIFRAC(directory_data+'/Atmosphere/IFRAC_Atlantic_year_2900-3800_month_1-12_QE.nc')

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('First EOF, quadratic detrending, lowpass filter 120 months', fontsize=15)
    
c1 = axs[0,0].contourf(lon_sst, lat_sst, -EOF_SST_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend='both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].contour(
    lon_ice, lat_ice,
    np.mean(IFRAC_month_forward, axis=0),
    levels=[0.195],          # or your exact level
    colors='black',
    linewidths=2,
    transform=ccrs.PlateCarree()
)
axs[0,0].set_title('a) First EOF SST - PI$^{\mathrm{on}}_{\mathrm{QE}}$ (var.ex. = '+str(int(VAR_E1_SST[0]*100))+'%)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_ylim(10,70)
axs[0,0].set_xlim(-90,20)

c2 = axs[0,1].contourf(lon_sst, lat_sst, EOF_SST_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
axs[0,1].contour(
    lon_ice, lat_ice,
    np.mean(IFRAC_month_backward, axis=0),
    levels=[0.195],          # or your exact level
    colors='black',
    linewidths=2,
    transform=ccrs.PlateCarree()
)#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) First EOF SST - PI$^{\mathrm{off}}_{\mathrm{QE}}$ (var.ex. = '+str(int(VAR_E2_SST[0]*100))+'%)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_ylim(10,70)
axs[0,1].set_xlim(-90,20)

c1 = axs[1,0].contourf(lon_temp, lat_temp, EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) First EOF temperature (100-300m) - PI$^{\mathrm{on}}_{\mathrm{QE}}$ (var.ex. = '+str(int(VAR_E1[0]*100))+'%)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_ylim(10,70)
axs[1,0].set_xlim(-90,20)

c2 = axs[1,1].contourf(lon_temp, lat_temp, EOF_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) First EOF temperature (100-300m) - PI$^{\mathrm{off}}_{\mathrm{QE}}$ (var.ex. = '+str(int(VAR_E2[0]*100))+'%)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_ylim(10,70)
axs[1,1].set_xlim(-90,20)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'EOF1_TEMP_depthaveraged_SST_ATLANTIC_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show()   

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('EOF patterns AMOC on', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, EOF_E1_detrend1_none[0,:]*np.std(PC_E1_detrend1_none[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend='both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) Detrend 1, no lowpass filter')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[0,1].contourf(lon, lat, EOF_E1_detrend1[0,:]*np.std(PC_E1_detrend1[0,:]) , transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) Detrend 1, lowpass filter 120mo running mean')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1,0].contourf(lon, lat, EOF_E1_detrend2_none[0,:]*np.std(PC_E1_detrend2_none[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) Detrend 2, no lowpass filter')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[1,1].contourf(lon, lat, EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Detrend 2, lowpass filter 120 mo running mean')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'EOF1_TEMP_depthaveraged_SST_ATLANTIC_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
plt.suptitle('First EOF AMOC off', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, EOF_E2_detrend1_none[0,:]*np.std(PC_E2_detrend1_none[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend='both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) Detrend 1, no lowpass filter')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[0,1].contourf(lon, lat, -EOF_E2_detrend1[0,:]*np.std(PC_E2_detrend1[0,:]) , transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) Detrend 1, lowpass filter 120 mo running mean')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1,0].contourf(lon, lat, EOF_E2_detrend2_none[0,:]*np.std(PC_E2_detrend2_none[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-0.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) Detrend 2, no lowpass filter')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[1,1].contourf(lon, lat, EOF_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.5,0.5,21), extend = 'both', cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Detrend 2, lowpass filter 120 mo running mean')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'EOF1_TEMP_depthaveraged_SST_ATLANTIC_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show() 

#%% plot PC's

#Central moving average
def Moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window = 20

fig, axs = plt.subplots(1, 2, figsize=(14, 4))  

#plt.suptitle('First PC NAO pattern', fontsize=14)

axs[0].set_title('a) PI$^{on}_{QE}$')
axs[0].plot(time_E1_none - time_E1_none[0], PC_E1_detrend2_none[0,:], color='red', alpha = 1, label='TEMP detrend2')
axs[0].plot(time_E1_none - time_E1_none[0], PC_E1_detrend1_none[0,:], color='blue', alpha = 1, label='TEMP detrend1')
axs[0].plot(time_E1 - time_E1[0], PC_E1[0,:], color='orange', alpha = 1, label='TEMP detrend2 runmean 120mo')
axs[0].plot(time_E1 - time_E1[0], PC_E1_detrend1[0,:], color='green', alpha = 1, label='TEMP detrend1 runmean 120mo')
#axs[0].plot(time_E1[window//2 : -window//2 + 1] - time_E1[0], Moving_average(PC_E1[0,:], window), color='orange')
#axs[0].plot(time_E1 - time_E1[0], PC_E1_SST[0,:], color='blue', alpha = 0.3, label='SST')
#axs[0].plot(time_E1[window//2 : -window//2 + 1] - time_E1[0], Moving_average(PC_E1_SST[0,:], window), color='blue')
#axs[0].set_ylim(-0.2, 0.35)
axs[0].legend()

axs[1].set_title('b) PI$^{off}_{QE}$')
axs[1].plot(time_E2_none - time_E2_none[0], PC_E2_detrend2_none[0,:], color='red', alpha = 1, label='TEMP detend2')
axs[1].plot(time_E2_none - time_E2_none[0], PC_E2_detrend1_none[0,:], color='blue', alpha = 1, label='TEMP detrend1')
axs[1].plot(time_E2 - time_E2[0], PC_E2[0,:], color='orange', alpha = 1, label='TEMP detend2 runmean 120mo')
axs[1].plot(time_E2 - time_E2[0], PC_E2_detrend1[0,:], color='green', alpha = 1, label='TEMP detrend1 runmean 120mo')
#axs[1].plot(time_E2[window//2 : -window//2 + 1] - time_E2[0], Moving_average(PC_E2[0,:], window), color='orange')
#axs[1].plot(time_E2 - time_E2[0], PC_E2_SST[0,:], color='blue', alpha = 0.3, label='SST')
#axs[1].plot(time_E2[window//2 : -window//2 + 1] - time_E2[0], Moving_average(PC_E2_SST[0,:], window), color='blue')
#axs[1].set_ylim(-0.2, 0.35)
axs[1].legend()


#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4))  

#plt.suptitle('First PC NAO pattern', fontsize=14)

axs[0].set_title('a) PI$^{on}_{QE}$')
axs[0].plot(time_E1 - time_E1[0], PC_E1[0,:], color='orange', alpha = 1, label='TEMP')
axs[0].plot(time_E1 - time_E1[0], PC_E1_SST[0,:], color='blue', alpha = 1, label='SST')
#axs[0].set_ylim(-0.2, 0.35)
axs[0].legend()

axs[1].set_title('b) PI$^{off}_{QE}$')
axs[1].plot(time_E2 - time_E2[0], PC_E2[0,:], color='orange', alpha = 1, label='TEMP')
axs[1].plot(time_E2 - time_E2[0], PC_E2_SST[0,:], color='blue', alpha = 1, label='SST')
#axs[1].set_ylim(-0.2, 0.35)
axs[1].legend()
# %%
