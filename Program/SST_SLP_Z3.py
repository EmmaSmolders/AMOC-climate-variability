#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:08:54 2026

@author: 6008399

sea level pressure of equilibrium branches

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
#import xesmf as xe
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
fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch600_year_999-1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon_SLP             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat_SLP             = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_1_annual   = fh.variables['SLP'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch1500_year_1899-2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
SLP_2_annual   = fh.variables['SLP'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch2900_year_2900-3500.nc', 'r')

time3           = fh.variables['time'][400:500] #Model years
SLP_3_annual   = fh.variables['SLP'][400:500,:,:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SLP_month_1-12_branch3800_year_4199-4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
SLP_4_annual   = fh.variables['SLP'][:]   #Mean reference air SLPerature

fh.close()

#%% 2m temperatures

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach600_year_999-1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_1_annual   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach1500_year_1899-2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_2_annual   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach2900_year_2900-3500.nc', 'r')

time3           = fh.variables['time'][400:500] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_3_annual   = fh.variables['TS'][400:500,:,:]   #Mean reference air temperature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'2m_TEMP_month_1-12_brach3800_year_4199-4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
lon             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat             = fh.variables['lat'][:]  #Array of latitudes [degN]
TEMP_4_annual   = fh.variables['TS'][:]   #Mean reference air temperature

fh.close()
#%% SST

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

#Read in data
fh      = netcdf.Dataset(directory_data+'SST_month_1-12_branch_600_year_999-1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
lon_SST             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat_SST             = fh.variables['lat'][:]  #Array of latitudes [degN]
SST_1_annual   = fh.variables['SST'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SST_month_1-12_branch_1500_year_1899-2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
SST_2_annual   = fh.variables['SST'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SST_month_1-12_branch_2900_year_3299-3400.nc', 'r')

time3           = fh.variables['time'][:] #Model years
SST_3_annual   = fh.variables['SST'][:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'SST_month_1-12_branch_3800_year_4199-4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
SST_4_annual   = fh.variables['SST'][:]   #Mean reference air SLPerature

fh.close()

#%% Geopotential height

#Read in data
fh      = netcdf.Dataset(directory_data+'Geopotential_height_200hPa_month_1-12_branch600_year_999_1100.nc', 'r')

time1           = fh.variables['time'][:] #Model years
time_jan        = fh.variables['time_month'][::12]
lon_Z3             = fh.variables['lon'][:]  #Array of longitudes [degE]
lat_Z3             = fh.variables['lat'][:]  #Array of latitudes [degN]
Z3_1_annual   = fh.variables['Z3'][:]   #Mean reference air SLPerature
Z3_1_jan   = fh.variables['Z3_month'][::12,:,:]   #Mean reference air SLPerature

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'Geopotential_height_200hPa_month_1-12_branch1500_year_1899_2000.nc', 'r')

time2           = fh.variables['time'][:] #Model years
Z3_2_annual   = fh.variables['Z3'][:]   #Mean reference air SLPerature
Z3_2_jan   = fh.variables['Z3_month'][::12,:,:]

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'Geopotential_height_200hPa_month_1-12_branch2900_year_3299_3400.nc', 'r')

time3           = fh.variables['time'][:] #Model years
Z3_3_annual   = fh.variables['Z3'][:]   #Mean reference air SLPerature
Z3_3_jan   = fh.variables['Z3_month'][::12,:,:]

fh.close()

#------------------------------------------------------------------------------------------

fh      = netcdf.Dataset(directory_data+'Geopotential_height_200hPa_month_1-12_branch3800_year_4199_4300.nc', 'r')

time4           = fh.variables['time'][:] #Model years
Z3_4_annual   = fh.variables['Z3'][:]   #Mean reference air SLPerature
Z3_4_jan   = fh.variables['Z3_month'][::12,:,:]

fh.close()

#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(
    lon_SLP, lat_SLP, np.mean(SLP_1_annual, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(960, 1030, 21), cmap='PuOr', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
cbar1.set_label('SLP [hPa]', fontsize=12)
#cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) Annual SLP (PI$^{on}_{18}$)', fontsize=14)
axs[0].grid()

contourf2 = axs[1].contourf(
    lon_SLP, lat_SLP, np.mean(SLP_4_annual, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(980, 1030, 21), cmap='PuOr', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('SLP [hPa]', fontsize=12)
#cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) Annual SLP (PI$^{off}_{18}$)', fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()


#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(
    lon_SLP, lat_SLP, np.mean(SLP_4_annual, axis=0) - np.mean(SLP_1_annual, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
cbar1.set_label('SLP difference [hPa]', fontsize=12)
cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) Annual SLP (PI$^{off}_{18}$ - PI$^{on}_{18}$)', fontsize=14)
axs[0].grid()

for lat_i in range(0, len(lat_SLP), 3):
    for lon_i in range(0, len(lon_SLP), 3):
        p_value = Welch(SLP_1_annual[:, lat_i, lon_i], SLP_4_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0].scatter(
                lon_SLP[lon_i], lat_SLP[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

contourf2 = axs[1].contourf(
    lon_SLP, lat_SLP, np.mean(SLP_3_annual, axis=0) - np.mean(SLP_2_annual, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('SLP difference [hPa]', fontsize=12)
cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) Annual SLP (PI$^{off}_{45}$ - PI$^{on}_{45}$)', fontsize=14)
axs[1].grid()

for lat_i in range(0, len(lat_SLP), 3):
    for lon_i in range(0, len(lon_SLP), 3):
        p_value = Welch(SLP_2_annual[:, lat_i, lon_i], SLP_3_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1].scatter(
                lon_SLP[lon_i], lat_SLP[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()

#%% SLP with zonal mean subtracted

slp_diff_18 = np.mean(SLP_4_annual, axis=0) - np.mean(SLP_1_annual, axis=0)
slp_diff_18_prime = slp_diff_18 - np.mean(slp_diff_18, axis=1, keepdims=True)

slp_diff_45 = np.mean(SLP_3_annual, axis=0) - np.mean(SLP_2_annual, axis=0)
slp_diff_45_prime = slp_diff_45 - np.mean(slp_diff_45, axis=1, keepdims=True)

SLP1_prime = SLP_1_annual - np.mean(SLP_1_annual, axis=2, keepdims=True)
SLP4_prime = SLP_4_annual - np.mean(SLP_4_annual, axis=2, keepdims=True)

SLP2_prime = SLP_2_annual - np.mean(SLP_2_annual, axis=2, keepdims=True)
SLP3_prime = SLP_3_annual - np.mean(SLP_3_annual, axis=2, keepdims=True)

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(
    lon_SLP, lat_SLP, slp_diff_18_prime,
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-2, 2, 21), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
cbar1.set_label('SLP difference [hPa]', fontsize=12)
cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) (Zonal mean subtraced) Annual SLP (PI$^{off}_{18}$ - PI$^{on}_{18}$)', fontsize=14)
axs[0].grid()

for lat_i in range(0, len(lat_SLP), 3):
    for lon_i in range(0, len(lon_SLP), 3):
        p_value = Welch(SLP1_prime[:, lat_i, lon_i], SLP4_prime[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0].scatter(
                lon_SLP[lon_i], lat_SLP[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

contourf2 = axs[1].contourf(
    lon_SLP, lat_SLP, slp_diff_45_prime,
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-2, 2, 21), cmap='RdBu_r', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('SLP difference [hPa]', fontsize=12)
cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) (Zonal mean subtraced) Annual SLP (PI$^{off}_{45}$ - PI$^{on}_{45}$)', fontsize=14)
axs[1].grid()

for lat_i in range(0, len(lat_SLP), 3):
    for lon_i in range(0, len(lon_SLP), 3):
        p_value = Welch(SLP2_prime[:, lat_i, lon_i], SLP3_prime[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1].scatter(
                lon_SLP[lon_i], lat_SLP[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(
    lon_SLP, lat_SLP, np.mean(SLP1_prime, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
cbar1.set_label('SLP [hPa]', fontsize=12)
#cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) (Zonal mean subtraced) Annual SLP (PI$^{on}_{18}$)', fontsize=14)
axs[0].grid()

contourf2 = axs[1].contourf(
    lon_SLP, lat_SLP, np.mean(SLP4_prime, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('SLP [hPa]', fontsize=12)
#cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) (Zonal mean subtraced) Annual SLP (PI$^{off}_{18}$)', fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()


#%%

def ConverterField(index_break, field):
	"""Shifts field, where it starts at 0E and ends at 360E"""

	new_field	= ma.masked_all(shape(field))
	length_section	= len(field[0]) - index_break

	#Shift the first part
	new_field[:, :length_section] = field[:, index_break:]

	#Shift the last part
	new_field[:, length_section:] = field[:, :index_break] 

	return new_field

def LowCESMPlot(lon, lat, field):
	"""Returns 4 array's to plot on a global projection"""

	#Left of pole
	lon[lon > 180]	= lon[lon > 180] - 360.0

	lon_1		= lon[:, :160]
	lat_1		= lat[:, :160]
	field_1		= field[:, :160]

	#Right of pole
	lon_2		= lon[:, 159:]
	lat_2		= lat[:, 159:]
	field_2		= field[:, 159:]

	lat_3		= ma.masked_where(lon_2 > 0.0, lat_2)
	field_3		= ma.masked_where(lon_2 > 0.0, field_2)
	lon_3		= ma.masked_where(lon_2 > 0.0, lon_2)

	lat_2		= ma.masked_where(lon_2 < 0.0, lat_2)
	field_2		= ma.masked_where(lon_2 < 0.0, field_2)
	lon_2		= ma.masked_where(lon_2 < 0.0, lon_2)

	#To match at 40W
	index_1		= (fabs(lon[40] - 0.0)).argmin()

	lon_4		= ConverterField(index_1, lon)
	lat_4		= ConverterField(index_1, lat)
	field_4		= ConverterField(index_1, field)

	lon_4		= lon_4[:, 280:300]
	lat_4		= lat_4[:, 280:300]
	field_4		= field_4[:, 280:300]

	return lon_1, lat_1, field_1, lon_2, lat_2, field_2, lon_3, lat_3, field_3, lon_4, lat_4, field_4

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


#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
#Rescale the temperature plot
scale	= 2
cut_off	= 2

SST_plot_18			            = np.mean(SST_4_annual, axis=0) - np.mean(SST_1_annual, axis=0)
SST_plot_18[SST_plot_18 < -cut_off]	= (SST_plot_18[SST_plot_18 < -cut_off] - -cut_off) / scale - cut_off
SST_plot_18[SST_plot_18 > cut_off]	= (SST_plot_18[SST_plot_18 > cut_off] - cut_off) / scale + cut_off


SST_plot_45			            = np.mean(SST_3_annual, axis=0) - np.mean(SST_2_annual, axis=0)
SST_plot_45[SST_plot_45 < -cut_off]	= (SST_plot_45[SST_plot_45 < -cut_off] - -cut_off) / scale - cut_off
SST_plot_45[SST_plot_45 > cut_off]	= (SST_plot_45[SST_plot_45 > cut_off] - cut_off) / scale + cut_off

#-----------------------------------------------------------------------------------------

lon_1, lat_1, SST_1_plot_18, lon_2, lat_2, SST_2_plot_18, lon_3, lat_3, SST_3_plot_18, lon_4, lat_4, SST_4_plot_18	= LowCESMPlot(lon_SST, lat_SST, SST_plot_18)
lon_1, lat_1, SST_1_plot_45, lon_2, lat_2, SST_2_plot_45, lon_3, lat_3, SST_3_plot_45, lon_4, lat_4, SST_4_plot_45= LowCESMPlot(lon_SST, lat_SST, SST_plot_45)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()})

CS      = ax.contourf(lon_1, lat_1, SST_1_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax.contourf(lon_2, lat_2, SST_2_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax.contourf(lon_3, lat_3, SST_3_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax.contourf(lon_4, lat_4, SST_4_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())


divider = make_axes_locatable(ax)
ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)

cbar    = colorbar(CS, ticks = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], cax=ax_cb)
cbar.ax.set_yticklabels([-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10])
cbar.set_label('Sea surface temperature difference ($^{\circ}$C)')

ax.set_global()
ax.gridlines(zorder = 11)
ax.add_feature(cfeature.LAND, zorder=10)
ax.coastlines()

ax.set_title('a) Sea surface temperature')
show()

#%%

SST_diff_18 = np.mean(SST_4_annual, axis=0) - np.mean(SST_1_annual, axis=0)
SST_diff_45 = np.mean(SST_3_annual, axis=0) - np.mean(SST_2_annual, axis=0)

plt.figure()
plt.contourf(SST_diff_18 - SST_diff_45, cmap='RdBu_r', levels=np.linspace(-2, 2, 21), extend='both')
plt.colorbar()
plt.title('Diff 18 - diff 45 SST')

plt.figure()
plt.contourf(np.mean(SST_4_annual, axis=0) - np.mean(SST_3_annual, axis=0), cmap='RdBu_r', levels=np.linspace(-2, 2, 21), extend='both')
plt.colorbar()
plt.title('E4  - E3 SST')

plt.figure()
plt.contourf(np.mean(SST_1_annual, axis=0) - np.mean(SST_2_annual, axis=0), cmap='RdBu_r', levels=np.linspace(-2, 2, 21), extend='both')
plt.colorbar()
plt.title('E1  - E2 SST')

#%%

#Remove zonal mean to show Rossby wave trains (two methods are here actually. See if they really differ)
# time means
Z_18_on  = np.mean(Z3_1_jan, axis=0)
Z_18_off = np.mean(Z3_4_jan, axis=0)

# experiment difference
Z_diff_18 = Z_18_off - Z_18_on

# remove zonal mean
Z_diff_prime_18 = Z_diff_18 - np.mean(Z_diff_18, axis=1, keepdims=True)

Z_45_on  = np.mean(Z3_2_jan, axis=0)
Z_45_off = np.mean(Z3_3_jan, axis=0)

Z_diff_45 = Z_45_off - Z_45_on

Z_diff_prime_45 = Z_diff_45 - np.mean(Z_diff_45, axis=1, keepdims=True)

#Other method: remove zonal mean from each state first
Z_18_on_prime  = Z3_1_jan  - np.mean(Z3_1_jan, axis=2, keepdims=True)
Z_18_off_prime = Z3_4_jan - np.mean(Z3_4_jan, axis=2, keepdims=True)

Z_45_on_prime  = Z3_2_jan  - np.mean(Z3_2_jan, axis=2, keepdims=True)
Z_45_off_prime = Z3_3_jan - np.mean(Z3_3_jan, axis=2, keepdims=True)

# Then compute difference
Z_prime_diff_18 = Z_18_off_prime - Z_18_on_prime

#%%

Z_clim = np.mean(Z3_1_annual, axis=0)
Z_anom = Z3_1_annual - Z_clim
Z_anom_prime = Z_anom - np.mean(Z_anom, axis=2, keepdims=True)

#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(lon_Z3, lat_Z3, Z_diff_prime_18, transform=ccrs.PlateCarree(),levels=np.linspace(-50, 50, 21), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
#cbar1.set_label('Z3 difference [hPa]', fontsize=12)
#cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) Jan Z3 (PI$^{off}_{18}$ - PI$^{on}_{18}$)', fontsize=14)
axs[0].grid()

contourf2 = axs[1].contourf(
    lon_Z3, lat_Z3, Z_diff_prime_45,
    transform=ccrs.PlateCarree(), levels=np.linspace(-50, 50, 21), cmap='RdBu_r', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('Z3 difference [hPa]', fontsize=12)
#cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) Jan Z3 (PI$^{off}_{45}$ - PI$^{on}_{45}$)', fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(lon_Z3, lat_Z3, Z_18_on - np.mean(Z_18_on, axis=1, keepdims=True), transform=ccrs.PlateCarree(),levels=np.linspace(-100, 100, 21), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
#cbar1.set_label('Z3 difference [hPa]', fontsize=12)
#cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) Jan Z3 (PI$^{on}_{18}$)', fontsize=14)
axs[0].grid()
contourf2 = axs[1].contourf(
    lon_Z3, lat_Z3, Z_18_off - np.mean(Z_18_off, axis=1, keepdims=True),
    transform=ccrs.PlateCarree(), levels=np.linspace(-100, 100, 21), cmap='RdBu_r', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('Z3 difference [hPa]', fontsize=12)
#cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) Jan Z3 (PI$^{off}_{18}$)', fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})

for ax in axs:
    ax.coastlines()
    
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--')

    gl.top_labels = False
    gl.right_labels = False

contourf1 = axs[0].contourf(lon_Z3, lat_Z3, Z_45_on - np.mean(Z_45_on, axis=1, keepdims=True), transform=ccrs.PlateCarree(),levels=np.linspace(-100, 100, 21), cmap='RdBu_r', extend='both')
cbar1 = fig.colorbar(contourf1, ax=axs[0])
#cbar1.set_label('Z3 difference [hPa]', fontsize=12)
#cbar1.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[0].set_title('a) Jan Z3 (PI$^{on}_{45}$)', fontsize=14)
axs[0].grid()
contourf2 = axs[1].contourf(
    lon_Z3, lat_Z3, Z_45_off - np.mean(Z_45_off, axis=1, keepdims=True),
    transform=ccrs.PlateCarree(), levels=np.linspace(-100, 100, 21), cmap='RdBu_r', extend='both')
cbar2 = fig.colorbar(contourf2, ax=axs[1])
cbar2.set_label('Z3 difference [hPa]', fontsize=12)
#cbar2.set_ticks([-6, -4, -2, 0, 2, 4, 6])

#axs[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
#axs[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
#axs[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
#axs[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
axs[1].set_title('b) Jan Z3 (PI$^{off}_{45}$)', fontsize=14)
axs[1].grid()

plt.tight_layout()
plt.show()

#%%

Z_on_anom  = Z3_1_annual - np.mean(Z3_1_annual[0:50], axis=0)
Z_off_anom = Z3_4_annual - np.mean(Z3_4_annual[0:50], axis=0)

Z_on_mean  = np.mean(Z_on_anom, axis=0)
Z_off_mean = np.mean(Z_off_anom, axis=0)

Z_diff = Z_off_mean - Z_on_mean

test = Z_diff - np.mean(Z_diff, axis=1, keepdims=True)

plt.figure()
plt.contourf(lon_Z3, lat_Z3, test, cmap='RdBu_r')
plt.colorbar()

#%%

from cartopy.util import add_cyclic_point

#lon_2, TEMP_1_annual_mean    = PeriodicBoundaries3D(lon, lat, TEMP_1_annual)
#lon_2, TEMP_2_annual_mean    = PeriodicBoundaries3D(lon, lat, TEMP_2_annual)
#lon_2, TEMP_3_annual_mean    = PeriodicBoundaries3D(lon, lat, TEMP_3_annual)
#lon, TEMP_4_annual_mean    = PeriodicBoundaries3D(lon, lat, TEMP_4_annual)

temp18_diff = np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0)
temp45_diff = np.mean(TEMP_3_annual, axis=0) - np.mean(TEMP_2_annual, axis=0)

temp18_c, lon_c = add_cyclic_point(temp18_diff, coord=lon)
temp45_c, lon_c = add_cyclic_point(temp45_diff, coord=lon)

fig, axs = plt.subplots(
    2, 2, figsize=(12, 8),
    subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})

# -----------------------------
for ax in axs.flat:
    ax.coastlines()

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--')
    
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator([-60, -180, 60])

CS = axs[0,0].contourf(
    lon_c, lat, temp18_c,
    levels=np.arange(-9, 9.01, 0.25),
    extend='both', cmap='RdBu_r',
    transform=ccrs.PlateCarree())

CS = axs[0,1].contourf(
    lon_c, lat, temp45_c,
    levels=np.arange(-9, 9.01, 0.25),
    extend='both', cmap='RdBu_r',
    transform=ccrs.PlateCarree())

# =========================================================
CS      = axs[0,0].contourf(lon, lat, np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0), levels = np.arange(-9, 9.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())

cb = fig.colorbar(CS, ax=axs[0, 0], orientation='vertical', shrink=0.6)
cb.set_label('Temperature difference [$^\\circ$C]', fontsize=12)
cb.set_ticks([-9, -6, -3, 0, 3, 6, 9])
axs[0, 0].set_title(r'a) Annual 2m temperature (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=14)

for lat_i in range(0, TEMP_1_annual.shape[1], 3):
    for lon_i in range(0, TEMP_1_annual.shape[2], 3):
        p_value = Welch(TEMP_1_annual[:, lat_i, lon_i], TEMP_4_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0, 0].scatter(
                lon[lon_i], lat[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

CS      = axs[0,1].contourf(lon, lat, np.mean(TEMP_3_annual, axis=0) - np.mean(TEMP_2_annual, axis=0), levels = np.arange(-9, 9.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())

cb = fig.colorbar(CS, ax=axs[0, 1], orientation='vertical', shrink=0.6)
cb.set_label('Temperature difference [$^\\circ$C]', fontsize=12)
cb.set_ticks([-9, -6, -3, 0, 3, 6, 9])
axs[0, 1].set_title(r'b) Annual 2m temperature (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=14)


for lat_i in range(0, TEMP_2_annual.shape[1], 3):
    for lon_i in range(0, TEMP_2_annual.shape[2], 3):
        p_value = Welch(TEMP_2_annual[:, lat_i, lon_i], TEMP_3_annual[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[0, 1].scatter(
                lon[lon_i], lat[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

# cf = axs[1, 0].contourf(lon_SLP, lat_SLP, np.mean(SLP_4_annual, axis=0) - np.mean(SLP_1_annual, axis=0), transform=ccrs.PlateCarree(), levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
# cb = fig.colorbar(cf, ax=axs[1, 0], orientation='vertical', shrink=0.8)
# cb.set_label('SLP difference [hPa]', fontsize=12)
# cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
# axs[1, 0].set_title(r'c) Annual SLP (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=14)

# for lat_i in range(0, len(lat_SLP), 3):
#     for lon_i in range(0, len(lon_SLP), 3):
#         p_value = Welch(SLP_1_annual[:, lat_i, lon_i], SLP_4_annual[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             axs[1, 0].scatter(
#                 lon_SLP[lon_i], lat_SLP[lat_i],
#                 marker='o', edgecolor='k', s=6, facecolors='none',
#                 transform=ccrs.PlateCarree())

# cf = axs[1, 1].contourf(
#     lon_SLP, lat_SLP,
#     np.mean(SLP_3_annual, axis=0) - np.mean(SLP_2_annual, axis=0),
#     transform=ccrs.PlateCarree(),
#     levels=np.linspace(-6, 6, 21), cmap='RdBu_r', extend='both')
# cb = fig.colorbar(cf, ax=axs[1, 1], orientation='vertical', shrink=0.8)
# cb.set_label('SLP difference [hPa]', fontsize=12)
# cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
# axs[1, 1].set_title(r'd) Annual SLP (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=14)

# for lat_i in range(0, len(lat_SLP), 3):
#     for lon_i in range(0, len(lon_SLP), 3):
#         p_value = Welch(SLP_2_annual[:, lat_i, lon_i], SLP_3_annual[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             axs[1, 1].scatter(
#                 lon_SLP[lon_i], lat_SLP[lat_i],
#                 marker='o', edgecolor='k', s=6, facecolors='none',
#                 transform=ccrs.PlateCarree())

# # =========================================================            

# cf = axs[2, 0].contourf(
#     lon_SLP, lat_SLP,
#     slp_diff_18_prime,
#     transform=ccrs.PlateCarree(),
#     levels=np.linspace(-2, 2, 21), cmap='RdBu_r', extend='both')
# cb = fig.colorbar(cf, ax=axs[2, 0], orientation='vertical', shrink=0.8)
# cb.set_label('SLP difference [hPa]', fontsize=12)
# cb.set_ticks([-2, -1, 0, 1, 2])
# axs[2, 0].set_title(r'e) (Zonal mean removed) Annual SLP (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=14)

# for lat_i in range(0, len(lat_SLP), 3):
#     for lon_i in range(0, len(lon_SLP), 3):
#         p_value = Welch(SLP1_prime[:, lat_i, lon_i], SLP4_prime[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             axs[2, 0].scatter(
#                 lon_SLP[lon_i], lat_SLP[lat_i],
#                 marker='o', edgecolor='k', s=6, facecolors='none',
#                 transform=ccrs.PlateCarree())

# cf = axs[2, 1].contourf(
#     lon_SLP, lat_SLP,
#     slp_diff_45_prime,
#     transform=ccrs.PlateCarree(),
#     levels=np.linspace(-2, 2, 21), cmap='RdBu_r', extend='both')
# cb = fig.colorbar(cf, ax=axs[2, 1], orientation='vertical', shrink=0.8)
# cb.set_label('SLP difference [hPa]', fontsize=12)
# cb.set_ticks([-2, -1, 0, 1, 2])
# axs[2, 1].set_title(r'f) (Zonal mean removed) Annual SLP (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=14)

# for lat_i in range(0, len(lat_SLP), 3):
#     for lon_i in range(0, len(lon_SLP), 3):
#         p_value = Welch(SLP2_prime[:, lat_i, lon_i], SLP3_prime[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             axs[2, 1].scatter(
#                 lon_SLP[lon_i], lat_SLP[lat_i],
#                 marker='o', edgecolor='k', s=6, facecolors='none',
#                 transform=ccrs.PlateCarree())

# =========================================================

cf = axs[1, 0].contourf(
    lon_Z3, lat_Z3,
    np.mean(Z_18_off_prime, axis=0) - np.mean(Z_18_on_prime, axis=0),
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-60, 60, 21), cmap='RdBu_r', extend='both')
cb = fig.colorbar(cf, ax=axs[1, 0], orientation='vertical', shrink=0.6)
cb.set_label('Z200 difference [m]', fontsize=12)
axs[1, 0].set_title(r'c) (Zonal mean removed) Jan Z200 (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=14)

for lat_i in range(0, len(lat_Z3), 4):
    for lon_i in range(0, len(lon_Z3), 4):
        p_value = Welch(Z_18_on_prime[:, lat_i, lon_i], Z_18_off_prime[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1, 0].scatter(
                lon_Z3[lon_i], lat_Z3[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

cf = axs[1, 1].contourf(
    lon_Z3, lat_Z3,
    Z_diff_prime_45,
    transform=ccrs.PlateCarree(),
    levels=np.linspace(-60, 60, 11), cmap='RdBu_r', extend='both')
cb = fig.colorbar(cf, ax=axs[1, 1], orientation='vertical', shrink=0.6)
cb.set_label('Z200 difference [m]', fontsize=12)
axs[1, 1].set_title(r'd) (Zonal mean removed) Jan Z200 (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=14)

for lat_i in range(0, len(lat_Z3), 4):
    for lon_i in range(0, len(lon_Z3), 4):
        p_value = Welch(Z_45_on_prime[:, lat_i, lon_i], Z_45_off_prime[:, lat_i, lon_i])
        if p_value <= 0.95:
            axs[1, 1].scatter(
                lon_Z3[lon_i], lat_Z3[lat_i],
                marker='o', edgecolor='k', s=6, facecolors='none',
                transform=ccrs.PlateCarree())

plt.tight_layout()
plt.savefig(directory_figures + 'SST_SLP_Z3_branches.pdf')
plt.show()

# %%
