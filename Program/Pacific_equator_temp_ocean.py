#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:38:00 2026

@author: 6008399

Oceanic temperature Pacific equator -> investigate thermocline feedback

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
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

#%% Read in data

month_start = 1
month_end   = 12

region      = 'global'
    
if month_start == 12 and month_end == 14:
    MONTHS = 'DJF'
    
if month_start == 6 and month_end == 8:
    MONTHS = 'JJA'
    
if month_start == 1 and month_end == 12:
    MONTHS = 'annual'

lat1, lat2 = 110, 260

fh      = netcdf.Dataset(directory_data+'TEMP_SALT_DENS_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch600_999-1100.nc', 'r')

time_E1        = fh.variables['time'][:] #Model years
lon            = fh.variables['lon'][:]  #Array of longitudes [degE]
depth          = fh.variables['depth'][:]
TEMP_E1        = fh.variables['TEMP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()


fh      = netcdf.Dataset(directory_data+'TEMP_SALT_DENS_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch1500_1899-2000.nc', 'r')

time_E2       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
TEMP_E2        = fh.variables['TEMP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'TEMP_SALT_DENS_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch2900_3299-3400.nc', 'r')

time_E3       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
TEMP_E3        = fh.variables['TEMP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'TEMP_SALT_DENS_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch3800_4199-4300.nc', 'r')

time_E4       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
TEMP_E4        = fh.variables['TEMP'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

#%%
fh      = netcdf.Dataset(directory_data+'VEL_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch600_999-1100.nc', 'r')

time_E1        = fh.variables['time'][:] #Model years
lon            = fh.variables['lon'][:]  #Array of longitudes [degE]
depth          = fh.variables['depth'][:]
WVEL_E1        = fh.variables['WVEL'][:]   #Sea level pressure (av\eraged over months) 

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch1500_1899-2000.nc', 'r')

time_E2       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
WVEL_E2        = fh.variables['WVEL'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch2900_3299-3400.nc', 'r')

time_E3       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
WVEL_E3        = fh.variables['WVEL'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_Pacific_meridional_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_year_branch3800_4199-4300.nc', 'r')

time_E4       = fh.variables['time'][:] #Model years
lon                 = fh.variables['lon'][:]  #Array of longitudes [degE]
WVEL_E4        = fh.variables['WVEL'][:]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_999-1100_month_1-12_branch600.nc', 'r')

time_month_E1       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E1        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E1              = fh.variables['SST'][0:100*12, lat1:lat2, :]
TAUX_month_E1       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E1             = fh.variables['TAUX'][0:100*12, lat1:lat2, :]

fh.close()


fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_1899-2000_month_1-12_branch1500.nc', 'r')

time_month_E2       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E2        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E2              = fh.variables['SST'][0:100*12, lat1:lat2, :]
TAUX_month_E2       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E2             = fh.variables['TAUX'][0:100*12, lat1:lat2, :]

fh.close()

fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_3299-3400_month_1-12_branch2900.nc', 'r')

time_month_E3       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E3        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E3              = fh.variables['SST'][0:100*12, lat1:lat2, :]
TAUX_month_E3       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E3             = fh.variables['TAUX'][0:100*12, lat1:lat2, :]

fh.close()

fh      = netcdf.Dataset(directory_data+'SST_TAUX_Pacific_year_4199-4300_month_1-12_branch3800.nc', 'r')

time_month_E4       = fh.variables['time_month'][0:100*12] #Model years
lon                 = fh.variables['lon'][lat1:lat2, :]  #Array of longitudes [degE]
lat                 = fh.variables['lat'][lat1:lat2, :]  #Array of latitudes [degN]
SST_month_E4        = fh.variables['SST_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
SST_E4              = fh.variables['SST'][0:100*12, lat1:lat2, :]
TAUX_month_E4       = fh.variables['TAUX_month'][0:100*12, lat1:lat2, :]   #Sea level pressure (av\eraged over months) [hPa]
TAUX_E4             = fh.variables['TAUX'][0:100*12, lat1:lat2, :]

fh.close()

#%%

divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=3)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contourf(lon[0, :], depth, np.mean(TEMP_E4, axis=0) - np.mean(TEMP_E1, axis=0), levels=np.linspace(-1, 3, 17), cmap='RdBu_r', norm = divnorm, extend='both')
cbar = fig.colorbar(im1, ax=axs[0]) 
cbar.set_label('Temperature difference [$^\circ$C]') # Add colorbar for the first subplot
axs[0].set_title('a) Temperature difference ('+str(MONTHS)+', PI$^{off}_{18}$ - PI$^{on}_{18}$')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')

# Subplot 2: TEMP_E3 - TEMP_E2
im2 = axs[1].contourf(lon[0, :], depth, np.mean(TEMP_E3, axis=0) - np.mean(TEMP_E2, axis=0), levels=np.linspace(-1, 3, 17), cmap='RdBu_r', norm = divnorm, extend='both')
cbar = fig.colorbar(im2, ax=axs[1])  # Add colorbar for the second subplot
cbar.set_label('Temperature difference [$^\circ$C]')
axs[1].set_title('b) Temperature difference ('+str(MONTHS)+', PI$^{off}_{45}$ - PI$^{on}_{45}$')
axs[1].set_ylim(depth[-1], 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(directory_figures +'TEMP_differences_Pacific_5S_5N_CESM_branches_'+str(MONTHS)+'_ocean.pdf')
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contourf(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=np.linspace(-2, 27, 21), extend='both')#, cmap='RdBu_r', norm = divnorm, extend='both')
cbar = fig.colorbar(im1, ax=axs[0]) 
cbar.set_label('Temperature difference [$^\circ$C]') # Add colorbar for the first subplot
axs[0].set_title('a) AMOC on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.18Sv)')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')

# Subplot 2: TEMP_E3 - TEMP_E2
im2 = axs[1].contourf(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=np.linspace(-2, 27, 21), extend='both')#, cmap='RdBu_r', norm = divnorm, extend='both')
cbar = fig.colorbar(im2, ax=axs[1])  # Add colorbar for the second subplot
cbar.set_label('Temperature difference [$^\circ$C]')
axs[1].set_title('b) AMOC off ('+str(MONTHS)+', $\\overline{F_H}$ = 0.18Sv)')
axs[1].set_ylim(depth[-1], 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()  
#plt.savefig(directory_figures +'TEMP_differences_Pacific_5S_5N_CESM_branches_'+str(MONTHS)+'_ocean.pdf')
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contourf(lon[0, :], depth, np.mean(TEMP_E2, axis=0), levels=np.linspace(-2, 27, 21), extend='both')#, cmap='RdBu_r', norm = divnorm, extend='both')
cbar = fig.colorbar(im1, ax=axs[0]) 
cbar.set_label('Temperature difference [$^\circ$C]') # Add colorbar for the first subplot
axs[0].set_title('a) AMOC on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.45Sv)')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')

# Subplot 2: TEMP_E3 - TEMP_E2
im2 = axs[1].contourf(lon[0, :], depth, np.mean(TEMP_E3, axis=0), levels=np.linspace(-2, 27, 21), extend='both')#, cmap='RdBu_r', norm = divnorm, extend='both')
cbar = fig.colorbar(im2, ax=axs[1])  # Add colorbar for the second subplot
cbar.set_label('Temperature difference [$^\circ$C]')
axs[1].set_title('b) AMOC off ('+str(MONTHS)+', $\\overline{F_H}$ = 0.45Sv)')
axs[1].set_ylim(depth[-1], 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


#%% Look at specific isotherms

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[20], colors='black', label='20 isoterm')
im2 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[20], colors='black', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 
#im1 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[22], colors='blue', label='22 isoterm')
#im2 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[22], colors='blue', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 
#im1 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[23], colors='green', label='23 isoterm')
#im2 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[23], colors='green', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 
im1 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[24], colors='red', label='24 isoterm')
im2 = axs[0].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[24], colors='red', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 


legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='20°C isotherm'),
    #Line2D([0], [0], color='black', linestyle='--', label='20°C isotherm (TEMP_E4)'),
    #Line2D([0], [0], color='blue', linestyle='-', label='22°C isotherm'),
    #Line2D([0], [0], color='blue', linestyle='--', label='22°C isotherm (TEMP_E4)'),
    #Line2D([0], [0], color='green', linestyle='-', label='23°C isotherm'),
    #Line2D([0], [0], color='green', linestyle='--', label='23°C isotherm (TEMP_E4)'),
    Line2D([0], [0], color='red', linestyle='-', label='24°C isotherm'),
    #Line2D([0], [0], color='red', linestyle='--', label='24°C isotherm (TEMP_E4)')
]
axs[0].legend(handles=legend_elements, loc = 4, fontsize=10)

#cbar.set_label('Temperature difference [$^\circ$C]') # Add colorbar for the first subplot
axs[0].set_title('a) AMOC off minus on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.18Sv)')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_ylim(200, 0)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')
#axs[0].legend()

# Subplot 2: TEMP_E3 - TEMP_E2
im1 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E2, axis=0), levels=[20], colors='black', label='20 isoterm')
im2 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E3, axis=0), levels=[20], colors='black', linestyles='--')
#im1 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[22], colors='blue', label='22 isoterm')
#im2 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[22], colors='blue', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 
#im1 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[23], colors='green', label='23 isoterm')
#im2 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[23], colors='green', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 
im1 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E1, axis=0), levels=[24], colors='red', label='24 isoterm')
im2 = axs[1].contour(lon[0, :], depth, np.mean(TEMP_E4, axis=0), levels=[24], colors='red', linestyles='--')#cbar = fig.colorbar(im1, ax=axs[0]) 
axs[1].set_title('b) AMOC off minus on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.45Sv)')
axs[1].set_ylim(200, 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()
plt.savefig(directory_figures +'Isoterm_differences_Pacific_5S_5N_CESM_branches_'+str(MONTHS)+'_ocean.pdf')
plt.show()

#%%

# Ensure TEMP_E1 and TEMP_E4 are 2D arrays (depth, longitude) by averaging over time
TEMP_E1_mean = np.mean(TEMP_E1, axis=0)  # Average over time (axis=0)
TEMP_E4_mean = np.mean(TEMP_E4, axis=0)  # Average over time (axis=0)

TEMP_E2_mean = np.mean(TEMP_E2, axis=0)  # Average over time (axis=0)
TEMP_E3_mean = np.mean(TEMP_E3, axis=0)  # Average over time (axis=0)

plt.figure()
plt.contourf(TEMP_E4_mean)

#%%

# Find the depth difference for the 20°C isotherm
isotherm_level = 20  # Define the isotherm level (e.g., 20°C)

# Initialize arrays to store depths of the isotherm
depth_E1 = np.full_like(lon[0, :], np.nan, dtype=float)  
depth_E2 = np.full_like(lon[0, :], np.nan, dtype=float)
depth_E3 = np.full_like(lon[0, :], np.nan, dtype=float)
depth_E4 = np.full_like(lon[0, :], np.nan, dtype=float)  


for i in range(len(lon[0, :])):
    
    temp_col1 = TEMP_E1_mean[:, i]
    temp_col2 = TEMP_E2_mean[:, i]
    temp_col3 = TEMP_E3_mean[:, i]
    temp_col4 = TEMP_E4_mean[:, i]

    #valid = ~temp_col.mask
    temp1 = temp_col1.data
    temp2 = temp_col2.data
    temp3 = temp_col3.data
    temp4 = temp_col4.data
    z = depth

    if temp1.size < 2:
        depth_E1[i] = np.nan
        print('hoi')
    else:
        order1 = np.argsort(temp1)
        temp1 = temp1[order1]
        order2 = np.argsort(temp2)
        temp2 = temp2[order2]
        order3 = np.argsort(temp3)
        temp3 = temp3[order3]
        order4 = np.argsort(temp4)
        temp4 = temp4[order4]
        z1 = z[order1]
        z2 = z[order2]
        z3 = z[order3]
        z4 = z[order4]

        if isotherm_level < temp1.min() or isotherm_level > temp1.max():
            depth_E1[i] = np.nan
        else:
            depth_E1[i] = np.interp(isotherm_level, temp1, z1)
            
        if isotherm_level < temp2.min() or isotherm_level > temp2.max():
            depth_E2[i] = np.nan
        else:
            depth_E2[i] = np.interp(isotherm_level, temp2, z2)
            
        if isotherm_level < temp3.min() or isotherm_level > temp3.max():
            depth_E3[i] = np.nan
        else:
            depth_E3[i] = np.interp(isotherm_level, temp3, z3)
            
        if isotherm_level < temp4.min() or isotherm_level > temp4.max():
            depth_E4[i] = np.nan
        else:
            depth_E4[i] = np.interp(isotherm_level, temp4, z4)



plt.figure()
plt.plot(lon[0, :], depth_E4 - depth_E1, color='black', label='PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$')
plt.plot(lon[0, :], depth_E3 - depth_E2, color='red', label='PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$')
plt.title('Depth difference 20°C Isotherm ('+str(MONTHS)+')')
plt.xlabel('Longitude [$^\circ$E]')
plt.ylabel('Depth Difference [m]')
plt.xlim(135, 275)
plt.ylim(-5, 15)
plt.legend()
plt.grid()
plt.savefig(directory_figures + 'depth_diff_branches_20C_isotherm_'+str(MONTHS)+'.pdf')





#%%

fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Create a 1x2 subplot layout

axs[0].plot(lon[0, :], depth_E1, color='darkorange', label='PI$^{\mathrm{on}}_{18}$')
axs[0].plot(lon[0, :], depth_E2, color='darkgreen', label='PI$^{\mathrm{on}}_{45}$')
axs[0].set_title('a) 20°C isotherm ('+str(MONTHS)+', AMOC on)')
axs[0].set_ylim(200, 0)
axs[0].set_xlim(140, 275)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')
axs[0].legend()
axs[0].grid()

axs[1].plot(lon[0, :], depth_E4 - depth_E1, color='darkorange', label='PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$')
axs[1].plot(lon[0, :], depth_E3 - depth_E2, color='darkgreen', label='PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$')
axs[1].set_title('b) Depth difference 20°C isotherm ('+str(MONTHS)+')')
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth Difference [m]')
axs[1].set_xlim(140, 275)
axs[1].set_ylim(-5, 15)
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig(directory_figures +'depth_diff_branches_20C_isotherm_'+str(MONTHS)+'.pdf')
plt.show()

#%%

plt.figure()
plt.plot(lon[0,:], np.mean(TEMP_E1[:,0,:], axis=0), label='E1')
plt.plot(lon[0,:], np.mean(TEMP_E4[:,0,:], axis=0), label='E4')
plt.legend()

plt.figure()
plt.plot(lon[0,:], np.mean(TEMP_E2[:,0,:], axis=0), label='E2')
plt.plot(lon[0,:], np.mean(TEMP_E3[:,0,:], axis=0), label='E3')
plt.legend()


#%%

divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=3)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contourf(lon[0, :], depth, np.mean(WVEL_E4, axis=0) - np.mean(WVEL_E1, axis=0), levels=np.linspace(-0.0005, 0.0005, 17), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(im1, ax=axs[0]) 
cbar.set_label('WVEL difference [m/s') # Add colorbar for the first subplot
axs[0].set_title('a) AMOC off minus on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.18Sv)')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')

# Subplot 2: TEMP_E3 - TEMP_E2
im2 = axs[1].contourf(lon[0, :], depth, np.mean(WVEL_E3, axis=0) - np.mean(WVEL_E2, axis=0), levels=np.linspace(-0.0005, 0.0005, 17), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(im2, ax=axs[1])  # Add colorbar for the second subplot
cbar.set_label('WVEL difference [m/s]')
axs[1].set_title('b) AMOC off minus on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.45Sv)')
axs[1].set_ylim(depth[-1], 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()  
plt.savefig(directory_figures +'WVEL_differences_Pacific_5S_5N_CESM_branches_'+str(MONTHS)+'_ocean.pdf')
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contourf(lon[0, :], depth, np.mean(WVEL_E1, axis=0), levels=np.linspace(-0.001, 0.001, 21), extend='both', cmap='RdBu_r')
cbar = fig.colorbar(im1, ax=axs[0]) 
cbar.set_label('WVEL [m/s]')
axs[0].set_title('a) AMOC on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.18Sv)')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')

# Subplot 2: TEMP_E3 - TEMP_E2
im2 = axs[1].contourf(lon[0, :], depth, np.mean(WVEL_E4, axis=0), levels=np.linspace(-0.001, 0.001, 21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(im2, ax=axs[1])  # Add colorbar for the second subplot
cbar.set_label('WVEL [m/s]')
axs[1].set_title('b) AMOC off ('+str(MONTHS)+', $\\overline{F_H}$ = 0.18Sv)')
axs[1].set_ylim(depth[-1], 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # Create a 1x2 subplot layout

# Subplot 1: TEMP_E4 - TEMP_E1
im1 = axs[0].contourf(lon[0, :], depth, np.mean(WVEL_E2, axis=0), levels=np.linspace(-0.001, 0.001, 21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(im1, ax=axs[0]) 
cbar.set_label('WVEL [m/s]')
axs[0].set_title('a) AMOC on ('+str(MONTHS)+', $\\overline{F_H}$ = 0.45Sv)')
axs[0].set_ylim(depth[-1], 0)
axs[0].set_xlim(110, 280)
axs[0].set_xlabel('Longitude [$^\circ$E]')
axs[0].set_ylabel('Depth [m]')

# Subplot 2: TEMP_E3 - TEMP_E2
im2 = axs[1].contourf(lon[0, :], depth, np.mean(WVEL_E3, axis=0), levels=np.linspace(-0.001, 0.001, 21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(im2, ax=axs[1])  # Add colorbar for the second subplot
cbar.set_label('WVEL [m/s]')
axs[1].set_title('b) AMOC off ('+str(MONTHS)+', $\\overline{F_H}$ = 0.45Sv)')
axs[1].set_ylim(depth[-1], 0)
axs[1].set_xlim(110, 280)
axs[1].set_xlabel('Longitude [$^\circ$E]')
axs[1].set_ylabel('Depth [m]')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
