#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:48:28 2025

@author: 6008399

Walker circulation using vertical velocities

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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

month_start = 1
month_end   = 12

region      = 'global'
    
if month_start == 12 and month_end == 14:
    MONTHS = 'DJF'
    
if month_start == 6 and month_end == 8:
    MONTHS = 'JJA'
    
if month_start == 1 and month_end == 12:
    MONTHS = 'annual'
 
#Read in data
fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch600_year_999_1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_1          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_1          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_1          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch1500_year_1899_2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_2          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_2          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_2          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch2900_year_3299_3400.nc', 'r')

time3           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_3          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_3          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_3          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch3800_year_4199_4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_4          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_4          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_4          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

#%% Air temperature 

fh      = netcdf.Dataset(directory_data+'Air_TEMP_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch600_year_999_1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
TEMP_1          = fh.variables['T'][:]    #Annual mean meridional mean zonal velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'Air_TEMP_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch1500_year_1899_2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
TEMP_2          = fh.variables['T'][:]    #Annual mean meridional mean zonal velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'Air_TEMP_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch2900_year_3299_3400.nc', 'r')

time3           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
TEMP_3          = fh.variables['T'][:]    #Annual mean meridional mean zonal velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'Air_TEMP_meridional_mean_5S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch3800_year_4199_4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
TEMP_4          = fh.variables['T'][:]    #Annual mean meridional mean zonal velocity

fh.close()

#%%

# Example data
x = lon  # Longitude (x-direction)
y = lev  # Pressure levels (decreasing)
u1 = np.mean(UVEL_1, axis=0) 
w1 = np.mean(WVEL_1, axis=0)  

u2 = np.mean(UVEL_2, axis=0) 
w2 = np.mean(WVEL_2, axis=0)  

u3 = np.mean(UVEL_3, axis=0) 
w3 = np.mean(WVEL_3, axis=0)  

u4 = np.mean(UVEL_4, axis=0) 
w4 = np.mean(WVEL_4, axis=0)  

# Reverse y and corresponding data to make y strictly increasing
y = y[::-1]
u1 = u1[::-1, :]
w1 = w1[::-1, :]

u2 = u2[::-1, :]
w2 = w2[::-1, :]

u3 = u3[::-1, :]
w3 = w3[::-1, :]

u4 = u4[::-1, :]
w4 = w4[::-1, :]

# Create a 2D grid of x and y
x_grid, y_grid = np.meshgrid(x, y)

# Interpolate u and w onto a uniform grid (if needed)
x_uniform = np.linspace(x.min(), x.max(), 100)  # Uniform longitude grid
y_uniform = np.linspace(y.min(), y.max(), 50)  # Uniform pressure grid
x_uniform_grid, y_uniform_grid = np.meshgrid(x_uniform, y_uniform)

u_interp1 = griddata((x_grid.flatten(), y_grid.flatten()), u1.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')
w_interp1 = griddata((x_grid.flatten(), y_grid.flatten()), w1.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')

u_interp2 = griddata((x_grid.flatten(), y_grid.flatten()), u2.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')
w_interp2 = griddata((x_grid.flatten(), y_grid.flatten()), w2.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')

u_interp3 = griddata((x_grid.flatten(), y_grid.flatten()), u3.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')
w_interp3 = griddata((x_grid.flatten(), y_grid.flatten()), w3.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')

u_interp4 = griddata((x_grid.flatten(), y_grid.flatten()), u4.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')
w_interp4 = griddata((x_grid.flatten(), y_grid.flatten()), w4.flatten(), (x_uniform_grid, y_uniform_grid), method='linear')

#%% UVEL

fig, axs = plt.subplots(2, 2, figsize=(12, 6))  

#axs[0,0].streamplot(x_uniform_grid, y_uniform_grid, u_interp1, w_interp1 * 10000, color='black', linewidth=1)

contourf1 = axs[0,0].contourf(
    lon, lev, np.mean(UVEL_1, axis=0),
    levels=np.linspace(-20, 20, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0,0])
cbar1.set_label('UVEL [m/s]', fontsize=12)
#cbar1.set_ticks([-9, -6, -3, 0, 3, 6, 9])

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,0].set_xticks(xtick_positions)
axs[0,0].set_xticklabels(xtick_labels)

axs[0,0].set_xlim(lon[0], lon[-1])
axs[0,0].set_ylim(1000, 100)
#axs[0,0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0,0].set_title('a) AMOC on ('+str(MONTHS)+', $F_H$ = 0.18Sv)', fontsize=14)

contourf2 = axs[0,1].contourf(
    lon, lev, (np.mean(UVEL_4, axis=0)), 
    levels=np.linspace(-20, 20, 41), cmap='seismic', extend='both')

cbar2 = fig.colorbar(contourf2, ax=axs[0,1])
cbar2.set_label('UVEL [m/s]', fontsize=12)
#cbar2.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[0,1].set_ylim(1000, 100)
axs[0,1].set_xlim(lon[0], lon[-1])
#axs[0,1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0,1].set_title('b) AMOC off ('+str(MONTHS)+', $F_H$ = 0.18Sv)', fontsize=14)

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,1].set_xticks(xtick_positions)
axs[0,1].set_xticklabels(xtick_labels)

#axs[0,1].streamplot(x_uniform_grid, y_uniform_grid, u_interp4, w_interp4 * 10000, color='black', linewidth=1)

#axs[1,0].streamplot(x_uniform_grid, y_uniform_grid, u_interp2, w_interp2 * 10000, color='black', linewidth=1)

contourf1 = axs[1,0].contourf(
    lon, lev, np.mean(UVEL_2, axis=0),
    levels=np.linspace(-20, 20, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[1,0])
cbar1.set_label('UVEL [m/s]', fontsize=12)
#cbar1.set_ticks([-9, -6, -3, 0, 3, 6, 9])

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,0].set_xticks(xtick_positions)
axs[1,0].set_xticklabels(xtick_labels)

axs[1,0].set_xlim(lon[0], lon[-1])
axs[1,0].set_ylim(1000, 100)
#axs[1,0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1,0].set_title('c) AMOC on ('+str(MONTHS)+', $F_H$ = 0.45Sv)', fontsize=14)

contourf2 = axs[1,1].contourf(
    lon, lev, (np.mean(UVEL_3, axis=0)), 
    levels=np.linspace(-20, 20, 41), cmap='seismic', extend='both')

cbar2 = fig.colorbar(contourf2, ax=axs[1,1])
cbar2.set_label('UVEL [m/s]', fontsize=12)
#cbar2.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[1,1].set_ylim(1000, 100)
axs[1,1].set_xlim(lon[0], lon[-1])
#axs[1,1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1,1].set_title('d) AMOC off ('+str(MONTHS)+', $F_H$ = 0.45Sv)', fontsize=14)

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,1].set_xticks(xtick_positions)
axs[1,1].set_xticklabels(xtick_labels)

#axs[1,1].streamplot(x_uniform_grid, y_uniform_grid, u_interp3, w_interp3 * 10000, color='black', linewidth=1)


plt.tight_layout()
#plt.savefig(directory_figures + 'UVEL_velocity_OFF_ON_forcing_018_045Sv_'+str(MONTHS)+'.pdf')
plt.show()

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.25)

axs = np.empty((2, 2), dtype=object)
axs[0, 0] = fig.add_subplot(gs[0, 0])
axs[0, 1] = fig.add_subplot(gs[0, 1])
axs[1, 0] = fig.add_subplot(gs[1, 0])
axs[1, 1] = fig.add_subplot(gs[1, 1])

# =========================
# Row 1
# =========================
contourf1 = axs[0,0].contourf(
    lon, lev, np.mean(UVEL_4, axis=0) - np.mean(UVEL_1, axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0,0])
cbar1.set_label('UVEL difference [m/s]', fontsize=12)
cbar1.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[0,0].set_ylim(1000, 100)
axs[0,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0,0].set_title(r'a) Zonal velocity difference ('+str(MONTHS)+r', PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=14)
axs[0,0].set_xlim(lon[0], lon[-1])

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,0].set_xticks(xtick_positions)
axs[0,0].set_xticklabels(xtick_labels)

skip_y = 3      # vertical spacing
skip_x = 10     # longitude spacing 

Q1 = axs[0,0].quiver(
    x_uniform_grid[::skip_y, ::skip_x],
    y_uniform_grid[::skip_y, ::skip_x],
    u_interp1[::skip_y, ::skip_x],
    np.zeros_like(u_interp1[::skip_y, ::skip_x]),
    scale=150,
    color='k')

axs[0,0].quiverkey(
    Q1,
    X=0.1, Y=-.17,          # position (relative to axis)
    U=10.0,                  # reference value (1 m/s)
    label='10 m/s',
    labelpos='E', 
    fontproperties={'size': 11})

contourf2 = axs[0,1].contourf(
    lon, lev, np.mean(UVEL_3, axis=0) - np.mean(UVEL_2, axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[0,1])
cbar2.set_label('UVEL difference [m/s]', fontsize=12)
cbar2.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[0,1].set_ylim(1000, 100)
axs[0,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0,1].set_title(r'b) Zonal velocity difference ('+str(MONTHS)+r', PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=14)
axs[0,1].set_xlim(lon[0], lon[-1])

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,1].set_xticks(xtick_positions)
axs[0,1].set_xticklabels(xtick_labels)

Q2 = axs[0,1].quiver(
    x_uniform_grid[::skip_y, ::skip_x],
    y_uniform_grid[::skip_y, ::skip_x],
    u_interp2[::skip_y, ::skip_x],
    np.zeros_like(u_interp2[::skip_y, ::skip_x]),
    scale=150,
    color='k')

axs[0,1].quiverkey(
    Q2,
    X=0.1, Y=-.17,          # position (relative to axis)
    U=10.0,                  # reference value (1 m/s)
    label='10 m/s',
    labelpos='E', 
    fontproperties={'size': 11})

# =========================
# Row 2
# =========================
contourf3 = axs[1,0].contourf(
    lon, lev, -np.mean(WVEL_4, axis=0) - -np.mean(WVEL_1, axis=0),
    levels=np.linspace(-0.02, 0.02, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[1,0])
cbar3.set_label('Omega difference [Pa/s, inverted]', fontsize=12)
cbar3.set_ticks([-0.02, -0.01, 0, 0.01, 0.02])
axs[1,0].set_ylim(1000, 100)
axs[1,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1,0].set_title(r'c) Vertical velocity difference ('+str(MONTHS)+r', PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=14)
axs[1,0].set_xlim(lon[0], lon[-1])

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,0].set_xticks(xtick_positions)
axs[1,0].set_xticklabels(xtick_labels)

skip_y = 5      # vertical spacing
skip_x = 6     # longitude spacing 

Q3 = axs[1,0].quiver(
    x_uniform_grid[::skip_y, ::skip_x],
    y_uniform_grid[::skip_y, ::skip_x],
    np.zeros_like(w_interp1[::skip_y, ::skip_x]),      # no horizontal component
    -w_interp1[::skip_y, ::skip_x] * 1000,             # vertical component
    scale=500,
    color='k')

contourf4 = axs[1,1].contourf(
    lon, lev, -np.mean(WVEL_3, axis=0) - -np.mean(WVEL_2, axis=0),
    levels=np.linspace(-0.02, 0.02, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[1,1])
cbar4.set_label('Omega difference [Pa/s, inverted]', fontsize=12)
cbar4.set_ticks([-0.02, -0.01, 0, 0.01, 0.02])
axs[1,1].set_ylim(1000, 100)
axs[1,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1,1].set_title(r'd) Vertical velocity difference ('+str(MONTHS)+r', PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=14)
axs[1,1].set_xlim(lon[0], lon[-1])

Q4 = axs[1,1].quiver(
    x_uniform_grid[::skip_y, ::skip_x],
    y_uniform_grid[::skip_y, ::skip_x],
    np.zeros_like(w_interp2[::skip_y, ::skip_x]),      # no horizontal component
    w_interp2[::skip_y, ::skip_x] * 1000,             # vertical component
    scale=500,
    color='k', 
    angles='xy')

#text next to quiver key
fig.text(
    0.59, 0.04,   
    r'$0.03\ \mathrm{Pa\ s^{-1}}$',
    ha='left',
    va='center')

fig.text(
    0.16, 0.04,  
    r'$0.03\ \mathrm{Pa\ s^{-1}}$',
    ha='left',
    va='center')

axs[1,0].quiverkey(Q3, 0.1, -0.17, 30, ' ', labelpos='E', transform=fig.transFigure, angle=90)
axs[1,1].quiverkey(Q4, 0.1, -0.17, 30, ' ', labelpos='E', transform=fig.transFigure, angle=90)

xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,1].set_xticks(xtick_positions)
axs[1,1].set_xticklabels(xtick_labels)

#plt.subplots_adjust(bottom=0.5)
plt.savefig(
    directory_figures + 'UVEL_WVEL_velocity_diff_OFF_ON_forcing_018_045Sv_' + str(MONTHS) + '_5S_5N.pdf',
    dpi=300)
plt.show()


 #%% Air temperature

fig, axs = plt.subplots(1, 2, figsize=(14, 4))  

contourf1 = axs[0].contourf(
    lon, lev, np.mean(TEMP_4, axis=0) - np.mean(TEMP_1, axis=0),
    levels=np.linspace(-1.5, 1.5, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
cbar1.set_label('Temperature difference [K]', fontsize=12)
#cbar1.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0].set_title('a) AMOC off - on ('+str(MONTHS)+', $F_H$ = 0.18Sv)', fontsize=14)
axs[0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0].set_xticks(xtick_positions)
axs[0].set_xticklabels(xtick_labels)


contourf2 = axs[1].contourf(
    lon, lev, (np.mean(TEMP_3, axis=0) - np.mean(TEMP_2, axis=0)), 
    levels=np.linspace(-1.5, 1.5, 41), cmap='seismic', extend='both')

cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('Temperature difference [K]', fontsize=12)
#cbar2.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1].set_title('b) AMOC off - on ('+str(MONTHS)+', $F_H$ = 0.45Sv)', fontsize=14)
axs[1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1].set_xticks(xtick_positions)
axs[1].set_xticklabels(xtick_labels)

plt.tight_layout()
plt.savefig(directory_figures + 'Air_temp_5S_5N_diff_OFF_ON_forcing_018_045Sv_'+str(MONTHS)+'.pdf')
plt.show()

#%%

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_1-12_QE_year_600_1500.nc', 'r')

time_forward           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_forward          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_forward          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_forward          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_1-12_QE_year_2900_3800.nc', 'r')

time_backward           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_backward          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_backward          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_backward          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_1-12_QE_year_300_600.nc', 'r')

time_forward2           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_forward2          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_forward2          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_forward2          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

fh      = netcdf.Dataset(directory_data+'VEL_meridional_mean_5S_5N_month_1-12_QE_year_3800_4100.nc', 'r')

time_backward2           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lev             = fh.variables['lev'][:]  #Pressure levels [hPa]
UVEL_backward2          = fh.variables['U'][:]    #Annual mean meridional mean zonal velocity
VVEL_backward2          = fh.variables['V'][:]    #Annual mean meridional mean meridional velocity
WVEL_backward2          = fh.variables['W'][:]    #Annual mean meridional mean vertical velocity

fh.close()

#%%

fig, axs = plt.subplots(5, 2, figsize=(12, 12))  

contourf1 = axs[0,0].contourf(
    lon, lev, np.mean(UVEL_backward[0:100], axis=0) - np.mean(UVEL_forward[800:-1], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0,0])
cbar2.set_label('Omega difference [Pa/s]', fontsize=12)
cbar2.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[0,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0, 0].set_title(
    'a) Year ('+str(int(time_backward[0]))+'-'+str(int(time_backward[100]))+') - ('+str(int(time_forward[800]))+'-'+str(int(time_forward[-1]))+') (annual)',
    fontsize=14)

axs[0,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,0].set_xticks(xtick_positions)
axs[0,0].set_xticklabels(xtick_labels)


contourf2 = axs[0,1].contourf(
    lon, lev, np.mean(UVEL_backward[200:300], axis=0) - np.mean(UVEL_forward[600:700], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[0,1])
cbar2.set_label('UVEL difference [m/s]', fontsize=12)
cbar2.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[0,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0, 1].set_title(
    'b) Year ('+str(int(time_backward[200]))+'-'+str(int(time_backward[300]))+') - ('+str(int(time_forward[600]))+'-'+str(int(time_forward[700]))+') (annual)',
    fontsize=14)
axs[0,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,1].set_xticks(xtick_positions)
axs[0,1].set_xticklabels(xtick_labels)

contourf3 = axs[1,0].contourf(
    lon, lev, np.mean(UVEL_backward[400:500], axis=0) - np.mean(UVEL_forward[400:500], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[1,0])
cbar3.set_label('UVEL difference [m/s]', fontsize=12)
cbar3.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[1,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1, 0].set_title(
    'c) Year ('+str(int(time_backward[400]))+'-'+str(int(time_backward[500]))+') - ('+str(int(time_forward[400]))+'-'+str(int(time_forward[500]))+') (annual)',
    fontsize=14)

axs[1,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,0].set_xticks(xtick_positions)
axs[1,0].set_xticklabels(xtick_labels)


contourf4 = axs[1,1].contourf(
    lon, lev, np.mean(UVEL_backward[600:700], axis=0) - np.mean(UVEL_forward[200:300], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[1,1])
cbar4.set_label('UVEL difference [m/s]', fontsize=12)
cbar4.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[1,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1, 1].set_title(
    'd) Year ('+str(int(time_backward[600]))+'-'+str(int(time_backward[700]))+') - ('+str(int(time_forward[200]))+'-'+str(int(time_forward[300]))+') (annual)',
    fontsize=14)
axs[1,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,1].set_xticks(xtick_positions)
axs[1,1].set_xticklabels(xtick_labels)

contourf3 = axs[2,0].contourf(
    lon, lev, np.mean(UVEL_backward[700:800], axis=0) - np.mean(UVEL_forward[100:200], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[2,0])
cbar3.set_label('UVEL difference [m/s]', fontsize=12)
cbar3.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[2,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[2,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[2, 0].set_title(
    'e) Year ('+str(int(time_backward[700]))+'-'+str(int(time_backward[800]))+') - ('+str(int(time_forward[100]))+'-'+str(int(time_forward[200]))+') (annual)',
    fontsize=14)

axs[2,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[2,0].set_xticks(xtick_positions)
axs[2,0].set_xticklabels(xtick_labels)


contourf4 = axs[2,1].contourf(
    lon, lev, np.mean(UVEL_backward[800::], axis=0) - np.mean(UVEL_forward[0:100], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[2,1])
cbar4.set_label('UVEL difference [m/s]', fontsize=12)
cbar4.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[2,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[2,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[2,1].set_title(
    'f) Year ('+str(int(time_backward[800]))+'-'+str(int(time_backward[-1]))+') - ('+str(int(time_forward[0]))+'-'+str(int(time_forward[100]))+') (annual)',
    fontsize=14)
axs[2,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[2,1].set_xticks(xtick_positions)
axs[2,1].set_xticklabels(xtick_labels)

contourf3 = axs[3,0].contourf(
    lon, lev, np.mean(UVEL_backward2[100:200], axis=0) - np.mean(UVEL_forward2[100:200], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[3,0])
cbar3.set_label('UVEL difference [m/s]', fontsize=12)
cbar3.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[3,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[3,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[3, 0].set_title(
    'g) Year ('+str(int(time_backward2[100]))+'-'+str(int(time_backward2[200]))+') - ('+str(int(time_forward2[100]))+'-'+str(int(time_forward2[200]))+') (annual)',
    fontsize=14)
axs[3,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[3,0].set_xticks(xtick_positions)
axs[3,0].set_xticklabels(xtick_labels)


contourf4 = axs[3,1].contourf(
    lon, lev, np.mean(UVEL_backward2[200:300], axis=0) - np.mean(UVEL_forward2[0:100], axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[3,1])
cbar4.set_label('UVEL difference [m/s]', fontsize=12)
cbar4.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[3,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[3,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[3,1].set_title(
    'f) Year ('+str(int(time_backward2[200]))+'-'+str(int(time_backward2[300]))+') - ('+str(int(time_forward2[0]))+'-'+str(int(time_forward2[100]))+') (annual)',
    fontsize=14)
axs[3,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[3,1].set_xticks(xtick_positions)
axs[3,1].set_xticklabels(xtick_labels)

contourf1 = axs[4,0].contourf(
    lon, lev, np.mean(UVEL_4, axis=0) - np.mean(UVEL_1, axis=0),
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[4,0])
cbar1.set_label('UVEL difference [m/s]', fontsize=12)
cbar1.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[4,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[4,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[4,0].set_title('g) ('+str(MONTHS)+', $F_H$ = 0.18Sv)', fontsize=14)
axs[4,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[4,0].set_xticks(xtick_positions)
axs[4,0].set_xticklabels(xtick_labels)


contourf2 = axs[4,1].contourf(
    lon, lev, (np.mean(UVEL_3, axis=0) - np.mean(UVEL_2, axis=0)), 
    levels=np.linspace(-3, 3, 41), cmap='seismic', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[4,1])
cbar2.set_label('UVEL difference [m/s]', fontsize=12)
cbar2.set_ticks([-3, -2, -1, 0, 1, 2, 3])
axs[4,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[4,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[4,1].set_title('h) ('+str(MONTHS)+', $F_H$ = 0.45Sv)', fontsize=14)
axs[4,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[4,1].set_xticks(xtick_positions)
axs[4,1].set_xticklabels(xtick_labels)

plt.tight_layout()
plt.savefig(directory_figures + 'UVEL_velocity_diff_OFF_ON_QE_annual.pdf')
plt.show()

#%%

fig, axs = plt.subplots(5, 2, figsize=(12, 12))  

contourf1 = axs[0,0].contourf(
    lon, lev, np.mean(WVEL_backward[0:100], axis=0) - np.mean(WVEL_forward[800:-1], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0,0])
cbar1.set_label('Omega difference [Pa/s]', fontsize=12)
cbar1.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[0,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0, 0].set_title(
    'a) Year ('+str(int(time_backward[0]))+'-'+str(int(time_backward[100]))+') - ('+str(int(time_forward[800]))+'-'+str(int(time_forward[-1]))+') (annual)',
    fontsize=14)
axs[0,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,0].set_xticks(xtick_positions)
axs[0,0].set_xticklabels(xtick_labels)


contourf2 = axs[0,1].contourf(
    lon, lev, np.mean(WVEL_backward[200:300], axis=0) - np.mean(WVEL_forward[600:700], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[0,1])
cbar2.set_label('Omega difference [Pa/s]', fontsize=12)
cbar2.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[0,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[0,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[0, 1].set_title(
    'b) Year ('+str(int(time_backward[200]))+'-'+str(int(time_backward[300]))+') - ('+str(int(time_forward[600]))+'-'+str(int(time_forward[700]))+') (annual)',
    fontsize=14)
axs[0,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[0,1].set_xticks(xtick_positions)
axs[0,1].set_xticklabels(xtick_labels)

contourf3 = axs[1,0].contourf(
    lon, lev, np.mean(WVEL_backward[400:500], axis=0) - np.mean(WVEL_forward[400:500], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[1,0])
cbar3.set_label('Omega difference [Pa/s]', fontsize=12)
cbar3.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[1,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1, 0].set_title(
    'c) Year ('+str(int(time_backward[400]))+'-'+str(int(time_backward[500]))+') - ('+str(int(time_forward[400]))+'-'+str(int(time_forward[500]))+') (annual)',
    fontsize=14)
axs[1,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,0].set_xticks(xtick_positions)
axs[1,0].set_xticklabels(xtick_labels)


contourf4 = axs[1,1].contourf(
    lon, lev, np.mean(WVEL_backward[600:700], axis=0) - np.mean(WVEL_forward[200:300], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[1,1])
cbar4.set_label('Omega difference [Pa/s]', fontsize=12)
cbar4.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[1,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[1,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[1, 1].set_title(
    'd) Year ('+str(int(time_backward[600]))+'-'+str(int(time_backward[700]))+') - ('+str(int(time_forward[200]))+'-'+str(int(time_forward[300]))+') (annual)',
    fontsize=14)
axs[1,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[1,1].set_xticks(xtick_positions)
axs[1,1].set_xticklabels(xtick_labels)

contourf3 = axs[2,0].contourf(
    lon, lev, np.mean(WVEL_backward[700:800], axis=0) - np.mean(WVEL_forward[100:200], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[2,0])
cbar3.set_label('Omega difference [Pa/s]', fontsize=12)
cbar3.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[2,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[2,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[2, 0].set_title(
    'e) Year ('+str(int(time_backward[700]))+'-'+str(int(time_backward[800]))+') - ('+str(int(time_forward[100]))+'-'+str(int(time_forward[200]))+') (annual)',
    fontsize=14)
axs[2,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[2,0].set_xticks(xtick_positions)
axs[2,0].set_xticklabels(xtick_labels)


contourf4 = axs[2,1].contourf(
    lon, lev, np.mean(WVEL_backward[800::], axis=0) - np.mean(WVEL_forward[0:100], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[2,1])
cbar4.set_label('Omega difference [Pa/s]', fontsize=12)
cbar4.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[2,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[2,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[2,1].set_title(
    'f) Year ('+str(int(time_backward[800]))+'-'+str(int(time_backward[-1]))+') - ('+str(int(time_forward[0]))+'-'+str(int(time_forward[100]))+') (annual)',
    fontsize=14)
axs[2,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[2,1].set_xticks(xtick_positions)
axs[2,1].set_xticklabels(xtick_labels)

contourf3 = axs[3,0].contourf(
    lon, lev, np.mean(WVEL_backward2[100:200], axis=0) - np.mean(WVEL_forward2[100:200], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar3 = fig.colorbar(contourf3, ax=axs[3,0])
cbar3.set_label('Omega difference [Pa/s]', fontsize=12)
cbar3.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[3,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[3,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[3, 0].set_title(
    'g) Year ('+str(int(time_backward2[100]))+'-'+str(int(time_backward2[200]))+') - ('+str(int(time_forward2[100]))+'-'+str(int(time_forward2[200]))+') (annual)',
    fontsize=14)
axs[3,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[3,0].set_xticks(xtick_positions)
axs[3,0].set_xticklabels(xtick_labels)


contourf4 = axs[3,1].contourf(
    lon, lev, np.mean(WVEL_backward2[200:300], axis=0) - np.mean(WVEL_forward2[0:100], axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar4 = fig.colorbar(contourf4, ax=axs[3,1])
cbar4.set_label('Omega difference [Pa/s]', fontsize=12)
cbar4.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[3,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[3,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[3,1].set_title(
    'f) Year ('+str(int(time_backward2[200]))+'-'+str(int(time_backward2[300]))+') - ('+str(int(time_forward2[0]))+'-'+str(int(time_forward2[100]))+') (annual)',
    fontsize=14)
axs[3,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[3,1].set_xticks(xtick_positions)
axs[3,1].set_xticklabels(xtick_labels)

contourf1 = axs[4,0].contourf(
    lon, lev, np.mean(WVEL_4, axis=0) - np.mean(WVEL_1, axis=0),
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[4,0])
cbar1.set_label('Omega difference [Pa/s]', fontsize=12)
cbar1.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[4,0].set_ylim(1000, 100)
#axs[0].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[4,0].set_ylabel('Pressure [hPa]', fontsize=12)
axs[4,0].set_title('e) ('+str(MONTHS)+', $F_H$ = 0.18Sv)', fontsize=14)
axs[4,0].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[4,0].set_xticks(xtick_positions)
axs[4,0].set_xticklabels(xtick_labels)


contourf2 = axs[4,1].contourf(
    lon, lev, (np.mean(WVEL_3, axis=0) - np.mean(WVEL_2, axis=0)), 
    levels=np.linspace(-0.03, 0.03, 41), cmap='seismic', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[4,1])
cbar2.set_label('Omega difference [Pa/s]', fontsize=12)
cbar2.set_ticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
axs[4,1].set_ylim(1000, 100)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
axs[4,1].set_ylabel('Pressure [hPa]', fontsize=12)
axs[4,1].set_title('f) ('+str(MONTHS)+', $F_H$ = 0.45Sv)', fontsize=14)
axs[4,1].set_xlim(lon[0], lon[-1])
xtick_positions = [30, 80, 130, 180, 230, 280, 320]  
xtick_labels = ['30$^\circ$E', '80$^\circ$E', '130$^\circ$E', '180$^\circ$E', '130$^\circ$W', '80$^\circ$W', '30$^\circ$W']
axs[4,1].set_xticks(xtick_positions)
axs[4,1].set_xticklabels(xtick_labels)


plt.tight_layout()
plt.savefig(directory_figures + 'WVEL_velocity_diff_OFF_ON_QE_annual.pdf')
plt.show()
