#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:06:17 2025

@author: 6008399

EOF ENSO plot

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
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
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

lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory + 'EOF_ENSO_SST_E1_moving_average_0_CESM_branch_year_999_1099.nc')
lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory + 'EOF_ENSO_SST_E2_moving_average_0_CESM_branch_year_1899_1999.nc')
lon, lat, eof_E3, time_E3, PC_E3, VAR_E3, EOF_E3		= ReadinData(directory + 'EOF_ENSO_SST_E3_moving_average_0_CESM_branch_year_3300_3399.nc')
lon, lat, eof_E4, time_E4, PC_E4, VAR_E4, EOF_E4		= ReadinData(directory + 'EOF_ENSO_SST_E4_moving_average_0_CESM_branch_year_4200_4299.nc')

#%% Take first EOF for NAO

EOF_TEMP_E1 = EOF_E1[0,:,:]
EOF_TEMP_E2 = EOF_E2[0,:,:]
EOF_TEMP_E3 = EOF_E3[0,:,:]
EOF_TEMP_E4 = EOF_E4[0,:,:]


#%%

#Align signs using correlation of PCs (first mode)
corr = np.corrcoef(EOF_TEMP_E1, EOF_TEMP_E2)[0,1]
if corr < 0:
    print('Switching signs')
    EOF_TEMP_E2 *= -1
    PC_E2[0,:] *= -1
    
#EOF_SLP_E4 = -EOF_SLP_E4
#PC_E4 = -PC_E4

EOF_TEMP_E2 = -EOF_TEMP_E2

#%%    

fig, axs = plt.subplots(2, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('First EOF monthly Pacific SST', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) First EOF PI$^{on}_{18}$ - monthly SSTs (var.ex. = '+str(int(VAR_E1[0]))+'%)')
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(0,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, EOF_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) First EOF PI$^{on}_{45}$ - monthly SSTs (var.ex. = '+str(int(VAR_E2[0]))+'%)')
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, EOF_TEMP_E4, transform=ccrs.PlateCarree(), levels = np.linspace(-0.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) First EOF PI$^{off}_{18}$ - monthly SSTs (var.ex. = '+str(int(VAR_E4[0]))+'%)')
axs[1,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, EOF_TEMP_E3, transform=ccrs.PlateCarree(), levels = np.linspace(-.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) First EOF PI$^{off}_{45}$ - montly SSTs (var.ex. = '+str(int(VAR_E3[0]))+'%)')
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
plt.savefig(directory_figures +'EOF_ENSO_SST_Pacific_moving_average_'+str(moving_average)+'_CESM_branches.pdf')
plt.show()    


#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0,0].contourf(lon, lat, EOF_TEMP_E1, transform=ccrs.PlateCarree(), levels = np.linspace(-0.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c1, ax=axs[0,0], orientation='horizontal')
axs[0,0].set_title('a) First EOF PI$^{\mathrm{on}}_{18}$ - monthly SSTs (var.ex. = '+str(int(VAR_E1[0]))+'%)')
axs[0,0].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)
axs[0,0].set_xlim(-80, 110)
axs[0,0].set_ylim(-20, 20)

c2 = axs[0,1].contourf(lon, lat, EOF_TEMP_E2, transform=ccrs.PlateCarree(), levels = np.linspace(-.03,0.03,21), cmap='RdBu_r')
fig.colorbar(c2, ax=axs[0,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) First EOF PI$^{\mathrm{on}}_{45}$ - monthly SSTs (var.ex. = '+str(int(VAR_E2[0]))+'%)')
axs[0,1].set_xticks(np.arange(-250,110,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)
axs[0,1].set_xlim(-80, 110)
axs[0,1].set_ylim(-20, 20)

c1 = axs[1,0].contourf(lon, lat, EOF_TEMP_E4 * np.std((PC_E4[0,:])) - EOF_TEMP_E1 * np.std((PC_E1[0,:])), transform=ccrs.PlateCarree(), levels = np.linspace(-0.0001,0.0001,21), cmap='RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[1,0], orientation='horizontal')
axs[1,0].set_title('c) Difference first EOF (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
axs[1,0].set_xticks(np.arange(-250,110,40), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)
axs[1,0].set_xlim(-80, 110)
axs[1,0].set_ylim(-20, 20)

c2 = axs[1,1].contourf(lon, lat, EOF_TEMP_E3 * np.std((PC_E3[0,:])) - EOF_TEMP_E2 * np.std((PC_E2[0,:])), transform=ccrs.PlateCarree(), levels = np.linspace(-.0001,0.0001,21), cmap='RdBu_r', extend = 'both')
fig.colorbar(c2, ax=axs[1,1], orientation='horizontal')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Difference first EOF (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
axs[1,1].set_xticks(np.arange(-250,110,40), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(-40,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)
axs[1,1].set_xlim(-80, 110)
axs[1,1].set_ylim(-20, 20)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'EOF_ENSO_SST_Pacific_diff_moving_average_'+str(moving_average)+'_CESM_QE.pdf')
plt.show()


#%% plot PC's

#Central moving average
def Moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window = 20

fig, axs = plt.subplots(1, 2, figsize=(14, 4))  

plt.suptitle('First PC ENSO pattern', fontsize=14)

axs[0].set_title('a) $F_H$ = 0.18Sv')
axs[0].plot(time_E1 - time_E1[0], PC_E1[0,:], color='orange', alpha = 0.3, label='AMOC on')
axs[0].plot(time_E1[window//2 : -window//2 + 1] - time_E1[0], Moving_average(PC_E1[0,:], window), color='orange')
axs[0].plot(time_E4 - time_E4[0], PC_E4[0,:], color='blue', alpha = 0.3, label='AMOC off')
axs[0].plot(time_E4[window//2 : -window//2 + 1] - time_E4[0], Moving_average(PC_E4[0,:], window), color='blue')
axs[0].set_ylim(-0.2, 0.35)
axs[0].legend()

axs[1].set_title('b) $F_H$ = 0.45Sv')
axs[1].plot(time_E2 - time_E2[0], PC_E2[0,:], color='orange', alpha = 0.3, label='AMOC on')
axs[1].plot(time_E2[window//2 : -window//2 + 1] - time_E2[0], Moving_average(PC_E2[0,:], window), color='orange')
axs[1].plot(time_E3 - time_E3[0], PC_E3[0,:], color='blue', alpha = 0.3, label='AMOC off')
axs[1].plot(time_E3[window//2 : -window//2 + 1] - time_E3[0], Moving_average(PC_E3[0,:], window), color='blue')
axs[1].set_ylim(-0.2, 0.35)
axs[1].legend()

#%%

print(std(PC_E1[0,:]))
print(std(PC_E2[0,:]))
print(std(PC_E3[0,:]))
print(std(PC_E4[0,:]))

print(std(PC_E4[0,:]) - std(PC_E1[0,:]))
print(std(PC_E3[0,:]) - std(PC_E2[0,:]))






# %%
