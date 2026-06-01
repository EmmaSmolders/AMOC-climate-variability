#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:59:37 2025

@author: 6008399

AMOC maximal plot forward and backward

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
from scipy import signal, stats
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
from scipy.interpolate import interp1d
from scipy.signal.windows import dpss

#Making pathway to folder with all data
directory = r'/Users/6008399/Documents/PhD/CESM_collapse/netcdf/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

from numpy.fft import fft, fftfreq

def power_spectrum_rednoise(data_series, time, surrogate=2000, max_lag=250):
    data_series = np.asarray(data_series, dtype=float)
    data_series = (data_series - np.mean(data_series)) / np.std(data_series, ddof=1)

    # --- AR(1) coefficient: robust estimate (lag-1 autocorr) ---
    a = np.corrcoef(data_series[:-1], data_series[1:])[0, 1]
    a = np.clip(a, -0.99, 0.99)  # avoid numerical issues if ~1

    var = np.var(data_series)
    b = np.sqrt((1.0 - a**2) * var)

    # --- Fourier spectrum of data ---
    freq_series = fft(data_series)
    freq_series = (np.real(freq_series)**2 + np.imag(freq_series)**2)

    # If time is evenly spaced, set dt properly:
    dt = np.median(np.diff(time)) if time is not None else 1.0
    freq = fftfreq(len(data_series), d=dt)

    # Keep non-negative frequencies
    pos = freq >= 0
    freq = freq[pos]
    freq_series = freq_series[pos]

    # --- Surrogates ---
    surrogate_fourier = np.zeros((surrogate, len(freq)))

    spin_up = 300
    for s in range(surrogate):
        dummy = np.zeros(len(data_series))
        signal = 0.0
        white = np.random.normal(0, 1, spin_up + len(dummy))

        for i in range(spin_up + len(dummy)):
            signal = a * signal + b * white[i]
            if i >= spin_up:
                dummy[i - spin_up] = signal

        f = fft(dummy)
        surrogate_fourier[s] = (np.real(f)**2 + np.imag(f)**2)[pos]

    cl_90 = np.percentile(surrogate_fourier, 90, axis=0)
    cl_95 = np.percentile(surrogate_fourier, 95, axis=0)
    cl_99 = np.percentile(surrogate_fourier, 99, axis=0)

    return freq, freq_series, cl_90, cl_95, cl_99, a

def TrendRemover(time, data, trend_type):
	"""Removes trend of choice"""
	
	rank = polyfit(time, data, trend_type)
	fitting = 0.0 
		
	for rank_i in range(len(rank)):
			
		fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))

	data -= fitting
	
	return data

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

#%%

region      = 'global'
    
#Read in data
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

#%% SSTs

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

#----------------------------------------------------------------------------------------
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


#%% AMOC strength at 26N

fh_transient = netcdf.Dataset(directory + 'AMOC_transport_depth_0-1000m_hysteresis.nc','r')
fh_branch1   = netcdf.Dataset(directory + 'AMOC_transport_depth_0-1000m_branch_0600.nc','r')
fh_branch2   = netcdf.Dataset(directory + 'AMOC_transport_depth_0-1000m_branch_1500.nc','r')
fh_branch4   = netcdf.Dataset(directory + 'AMOC_transport_depth_0-1000m_branch_2900.nc','r')
fh_branch5   = netcdf.Dataset(directory + 'AMOC_transport_depth_0-1000m_branch_3800.nc','r')

time_transient      = fh_transient.variables['time'][:]     #time in model years
time_branch1        = fh_branch1.variables['time'][:]       #time in model years
time_branch2        = fh_branch2.variables['time'][:]       #time in model years
time_branch4        = fh_branch4.variables['time'][:]       #time in model years
time_branch5        = fh_branch5.variables['time'][:]       #time in model years

AMOC_transient  = fh_transient.variables['Transport'][:] #Volume transport [Sv]
AMOC_branch1    = fh_branch1.variables['Transport'][:]   
AMOC_branch2    = fh_branch2.variables['Transport'][:]   
AMOC_branch4    = fh_branch4.variables['Transport'][:]   
AMOC_branch5    = fh_branch5.variables['Transport'][:]   

fh_transient.close()
fh_branch1.close()
fh_branch2.close()
fh_branch4.close()
fh_branch5.close()

#%% AMOC structure last 50 years of branch 1 and 2

fh = netcdf.Dataset(directory + 'AMOC_structure_year_1050-1100_branch_600.nc', 'r')

depth_amoc	= fh.variables['depth'][:] 	 / 1000.0	
lat_amoc	    = fh.variables['lat'][:] 		
AMOC_1	= np.mean(fh.variables['AMOC'][:], axis = 0)	

fh.close()

fh = netcdf.Dataset(directory + 'AMOC_structure_year_1950-2000_branch_1500.nc', 'r')

AMOC_2	= np.mean(fh.variables['AMOC'][:], axis = 0)	

fh.close()

fh = netcdf.Dataset(directory + 'AMOC_structure_year_3350-3400_branch_2900.nc', 'r')

depth	= fh.variables['depth'][:] 	 / 1000.0		
AMOC_3	= np.mean(fh.variables['AMOC'][:], axis = 0)	

fh.close()

fh = netcdf.Dataset(directory + 'AMOC_structure_year_4250-4300_branch_3800.nc', 'r')

AMOC_4	= np.mean(fh.variables['AMOC'][:], axis = 0)	

fh.close()

#%% FOV 

fh_fov_60N_transient = netcdf.Dataset(directory + 'FOV_60N_year_0_2200.nc','r')
fh_fov_26N_transient = netcdf.Dataset(directory + 'FOV_26N_year_0_2200.nc','r')
fh_fov_34S_transient = netcdf.Dataset(directory + 'FOV_34S_year_0_2200.nc','r')

fh_fov_60N_branch1 = netcdf.Dataset(directory + 'FOV_60N_year_600_1100_branch_600.nc','r')
fh_fov_26N_branch1 = netcdf.Dataset(directory + 'FOV_26N_year_600_1100_branch_600.nc','r')
fh_fov_34S_branch1 = netcdf.Dataset(directory + 'FOV_34S_year_600_1100_branch_600.nc','r')

fh_fov_60N_branch2 = netcdf.Dataset(directory + 'FOV_60N_year_1500_2000_branch_1500.nc','r')
fh_fov_26N_branch2 = netcdf.Dataset(directory + 'FOV_26N_year_1500_2000_branch_1500.nc','r')
fh_fov_34S_branch2 = netcdf.Dataset(directory + 'FOV_34S_year_1500_2000_branch_1500.nc','r')

fh_fov_60N_branch3 = netcdf.Dataset(directory + 'FOV_60N_year_2900_3051_branch_1650.nc','r')
fh_fov_26N_branch3 = netcdf.Dataset(directory + 'FOV_26N_year_2900_3051_branch_1650.nc','r')
fh_fov_34S_branch3 = netcdf.Dataset(directory + 'FOV_34S_year_2900_3051_branch_1650.nc','r')

FOV_60N_transient  = fh_fov_60N_transient.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_26N_transient  = fh_fov_26N_transient.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_34S_transient  = fh_fov_34S_transient.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])

FOV_60N_branch1  = fh_fov_60N_branch1.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_26N_branch1  = fh_fov_26N_branch1.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_34S_branch1  = fh_fov_34S_branch1.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])

FOV_60N_branch2  = fh_fov_60N_branch2.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_26N_branch2  = fh_fov_26N_branch2.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_34S_branch2  = fh_fov_34S_branch2.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])

FOV_60N_branch3  = fh_fov_60N_branch3.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_26N_branch3  = fh_fov_26N_branch3.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])
FOV_34S_branch3  = fh_fov_34S_branch3.variables['F_OV'][:] #Freshwater transport (overturning component [Sv])


#%% Figure 1 of revised paper (as function of Fh and including E3)

lat_idx_samba = 0
lat_idx_rapid = 190
lat_idx_60N   = 253

#%% MHT

def ReadinData(filename):

	fh = netcdf.Dataset(filename, 'r')

	lat	= fh.variables['lat'][:]		#Latitudes (degrees N)
	MHT	= fh.variables['MHT'][:]		#Meridional heat transport (PW)

	fh.close()

	return lat, MHT

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------	

year_start		= 999
year_end		= 1100
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_on	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_600.nc')
lat_ocn, MHT_ocn_on	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_600.nc')

#-----------------------------------------------------------------------------------------

year_start		= 4199
year_end		= 4300
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_off	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_3800.nc')
lat_ocn, MHT_ocn_off	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_3800.nc')

year_start		= 1899
year_end		= 2000
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_on_45	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_1500.nc')
lat_ocn, MHT_ocn_on_45	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_1500.nc')

#-----------------------------------------------------------------------------------------

year_start		= 3299
year_end		= 3400
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_off_45	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_2900.nc')
lat_ocn, MHT_ocn_off_45	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_2900.nc')

plt.figure()
plt.plot(lat_ocn, MHT_ocn_off)
plt.plot(lat_ocn, MHT_ocn_on)
plt.plot(lat_ocn, MHT_ocn_off_45)
plt.plot(lat_ocn, MHT_ocn_on_45)

#%%
#Interpolate the oceanic heat transport to the atmospheric grid
#Skip the first 8 as there is no oceanic heat transport on Antarctic (of course)
MHT_ocn_on_int		= interp1d(lat_ocn, MHT_ocn_on)(np.array(lat_atm[8:]))
MHT_ocn_off_int		= interp1d(lat_ocn, MHT_ocn_off)(np.array(lat_atm[8:]))

MHT_atm_on	= np.copy(MHT_tot_on)
MHT_atm_off	= np.copy(MHT_tot_off)
MHT_atm_on[8:]  -= MHT_ocn_on_int
MHT_atm_off[8:] -= MHT_ocn_off_int
#-----------------------------------------------------------------------------------------

MHT_ocn_on_int_45		= interp1d(lat_ocn, MHT_ocn_on_45)(np.array(lat_atm[8:]))
MHT_ocn_off_int_45		= interp1d(lat_ocn, MHT_ocn_off_45)(np.array(lat_atm[8:]))

MHT_atm_on_45	= np.copy(MHT_tot_on_45)
MHT_atm_off_45	= np.copy(MHT_tot_off_45)
MHT_atm_on_45[8:]  -= MHT_ocn_on_int_45
MHT_atm_off_45[8:] -= MHT_ocn_off_int_45

#%%

divnorm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=26)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

CS1 = axs[0,0].contourf(lat_amoc, depth_amoc, AMOC_1, cmap='RdBu_r', levels=np.linspace(-10, 26, 19), extend='both', norm = divnorm)
cbar = fig.colorbar(CS1)
cbar.set_label('Meridional streamfunction [Sv]',  fontsize=11)
axs[0,0].set_ylim(depth[-1], 0)
axs[0,0].set_title('a) AMOC on (PI$_{18}$)', fontsize=13)
axs[0,0].set_xlabel('Latitude [$^\circ$N]', fontsize=12)

CS2 = axs[0,1].contourf(lat_amoc, depth_amoc, AMOC_4, cmap='RdBu_r', levels=np.linspace(-10, 26, 19), extend='both', norm=divnorm)
cbar = fig.colorbar(CS2)
cbar.set_label('Meridional streamfunction [Sv]', fontsize=11)
axs[0,1].set_ylim(depth[-1], 0)
axs[0,1].tick_params(axis='both', which='major', labelsize=10)
axs[0,1].set_xlabel('Latitude [$^\circ$N]', fontsize=12)
axs[0,0].set_ylabel('Depth [km]', fontsize=12)
axs[0,1].set_title('b) AMOC off (PI$_{18}$)', fontsize=13)

CS1 = axs[1,0].contourf(lat_amoc, depth_amoc, AMOC_2, cmap='RdBu_r', levels=np.linspace(-10, 26, 19), extend='both', norm=divnorm)
cbar = fig.colorbar(CS1)
cbar.set_label('Meridional streamfunction [Sv]',  fontsize=11)
axs[1,0].set_ylim(depth[-1], 0)
axs[1,0].set_title('c) AMOC on (PI$_{45}$)', fontsize=13)
axs[1,0].set_xlabel('Latitude [$^\circ$N]', fontsize=12)

CS2 = axs[1,1].contourf(lat_amoc, depth_amoc, AMOC_3, cmap='RdBu_r', levels=np.linspace(-10, 26, 19), extend='both', norm=divnorm)
cbar = fig.colorbar(CS2)
cbar.set_label('Meridional streamfunction [Sv]', fontsize=11)
axs[1,1].set_ylim(depth[-1], 0)
axs[1,1].tick_params(axis='both', which='major', labelsize=10)
axs[1,1].set_xlabel('Latitude [$^\circ$N]', fontsize=12)
axs[1,0].set_ylabel('Depth [km]', fontsize=12)
axs[1,1].set_title('d) AMOC off (PI$_{45}$)', fontsize=13)

plt.tight_layout()
plt.savefig(directory_figures +'Overturning_equilibria.pdf')
plt.show()

#%%

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2)

axs[0,0] = fig.add_subplot(gs[0, 0])                             
axs[0,1] = fig.add_subplot(gs[0, 1])                            
ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson()) 
ax_d = fig.add_subplot(gs[1, 1])                            

axs[0,0].plot(time_transient[0:2200]*0.0003, AMOC_transient[0:2200], color = 'black', label='Forward quasi-equilibrium', linewidth=1)
axs[0,0].plot((4400 - time_transient[2201:4400])*0.0003, AMOC_transient[2201:4400].transpose(), color = 'grey', label='Backward quasi-equilibrium', linewidth=1)

axs[0,0].plot(time_branch1[0]*0.0003, np.mean(AMOC_branch1[350:500]), 'o', color = 'orange', label='PI$_{18}$', markersize=10)
axs[0,0].vlines(x = time_branch1[0]*0.0003, ymin=np.min(AMOC_branch1[350:500]), ymax= np.max(AMOC_branch1[350:500]), color='orange', linewidth=2)
axs[0,0].hlines(xmin = time_branch1[0]*0.0003 - 20*0.0003, xmax = time_branch1[0]*0.0003 + 20*0.0003, y=np.min(AMOC_branch1[350:500]), color='orange', linewidth=2)
axs[0,0].hlines(xmin = time_branch1[0]*0.0003 - 20*0.0003, xmax = time_branch1[0]*0.0003 + 20*0.0003, y=np.max(AMOC_branch1[350:500]), color='orange', linewidth=2)

axs[0,0].plot(time_branch2[0]*0.0003, np.mean(AMOC_branch4[350:500]), 'o', color = 'green', markersize=10)
axs[0,0].vlines(x = time_branch2[0]*0.0003, ymin=np.min(AMOC_branch4[350:500]), ymax= np.max(AMOC_branch4[350:500]), color='green', linewidth=2)
axs[0,0].hlines(xmin = time_branch2[0]*0.0003 - 20*0.0003, xmax = time_branch2[0]*0.0003 + 20*0.0003, y=np.min(AMOC_branch4[350:500]), color='green', linewidth=2)
axs[0,0].hlines(xmin = time_branch2[0]*0.0003 - 20*0.0003, xmax = time_branch2[0]*0.0003 + 20*0.0003, y=np.max(AMOC_branch4[350:500]), color='green', linewidth=2)

axs[0,0].plot(time_branch1[0]*0.0003, np.mean(AMOC_branch5[350:500]), 'o', color = 'orange', markersize=10)
axs[0,0].vlines(x = time_branch1[0]*0.0003, ymin=np.min(AMOC_branch5[350:500]), ymax= np.max(AMOC_branch5[350:500]), color='orange', linewidth=2)
axs[0,0].hlines(xmin = time_branch1[0]*0.0003 - 20*0.0003, xmax = time_branch1[0]*0.0003 + 20*0.0003, y=np.min(AMOC_branch5[350:500]), color='orange', linewidth=2)
axs[0,0].hlines(xmin = time_branch1[0]*0.0003 - 20*0.0003, xmax = time_branch1[0]*0.0003 + 20*0.0003, y=np.max(AMOC_branch5[350:500]), color='orange', linewidth=2)

axs[0,0].plot(time_branch2[0]*0.0003, np.mean(AMOC_branch2[350:500]), 'o', color = 'green', label='PI$_{45}$', markersize=10)
axs[0,0].vlines(x = time_branch2[0]*0.0003, ymin=np.min(AMOC_branch2[350:500]), ymax= np.max(AMOC_branch2[350:500]), color='green', linewidth=2)
axs[0,0].hlines(xmin = time_branch2[0]*0.0003 - 20*0.0003, xmax = time_branch2[0]*0.0003 + 20*0.0003, y=np.min(AMOC_branch2[350:500]), color='green', linewidth=2)
axs[0,0].hlines(xmin = time_branch2[0]*0.0003 - 20*0.0003, xmax = time_branch2[0]*0.0003 + 20*0.0003, y=np.max(AMOC_branch2[350:500]), color='green', linewidth=2)

axs[0,0].set_ylim(-2,35)
#axs[0].set_xlim(0,0.66)
axs[0,0].set_xlim(0,2200*0.0003)
#axs[0].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axs[0,0].tick_params(axis='both', which='major', labelsize=10)
axs[0,0].set_xlabel('Freshwater flux forcing F$_H$ [Sv]', fontsize=12)
#axs[0,0].set_xlabel('Time [model years]', fontsize=12)
axs[0,0].set_ylabel('Volume transport [Sv]', fontsize=12)
axs[0,0].set_title('a) AMOC strength at 26$^\circ$N', fontsize=13)
axs[0,0].grid()
axs[0,0].legend(fontsize=11)

CS_black = axs[0,1].contour(lat_amoc, depth_amoc, AMOC_1, colors='black', levels=np.linspace(0, 23, 5))
axs[0,1].clabel(CS_black, inline=True, fontsize=10, fmt='%1.1f')  

CS1 = axs[0,1].contourf(lat_amoc, depth_amoc, AMOC_4 - AMOC_1, cmap='RdBu_r', levels=np.linspace(-13, 13, 21), extend='both')
cbar = fig.colorbar(CS1)
cbar.set_label('AMOC difference [Sv]',  fontsize=11)
axs[0,1].set_ylim(depth[-1], 0)
axs[0,1].set_title('b) AMOC streamfunction (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)
axs[0,1].set_xticks(np.arange(-30, 90.1, 30))
axs[0,1].set_xticklabels(['30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])
axs[0,1].set_ylabel('Depth [km]', fontsize=12)

# =========================================================
CS      = ax_c.contourf(lon_1, lat_1, SST_1_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax_c.contourf(lon_2, lat_2, SST_2_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax_c.contourf(lon_3, lat_3, SST_3_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax_c.contourf(lon_4, lat_4, SST_4_plot_18, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())

cb = fig.colorbar(CS, ax=ax_c, orientation='horizontal', pad=0.05, fraction=0.05, shrink=1)
cb.set_label('SST difference [$^\\circ$C]', fontsize=11)
cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
ax_c.set_title(r'c) Annual SST (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)

# for lat_i in range(0, SST_1_annual.shape[1], 10):
#     for lon_i in range(0, SST_1_annual.shape[2], 10):
#         if SST_1_annual[:,lat_i, lon_i].all() is ma.masked:
#              continue
#         p_value = Welch(SST_1_annual[:, lat_i, lon_i], SST_4_annual[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             ax_c.scatter(
#                 lon_SST[lat_i, lon_i], lat_SST[lat_i, lon_i],
#                 marker='o', edgecolor='k', s=6, facecolors='none',
#                 transform=ccrs.PlateCarree())


sig_mask_18 = ma.masked_all((SST_1_annual.shape[1], SST_1_annual.shape[2]))

for lat_i in range(SST_1_annual.shape[1]):
    for lon_i in range(SST_1_annual.shape[2]):
        # skip masked ocean/land points
        if ma.getmaskarray(SST_1_annual[:, lat_i, lon_i]).all():
            continue
        if ma.getmaskarray(SST_4_annual[:, lat_i, lon_i]).all():
            continue

        p_value = Welch(SST_1_annual[:, lat_i, lon_i], SST_4_annual[:, lat_i, lon_i])

        # mark NON-significant points
        if p_value <= 0.95:
            sig_mask_18[lat_i, lon_i] = 1.0
            
_, _, sig1_18, _, _, sig2_18, _, _, sig3_18, _, _, sig4_18 = LowCESMPlot(
    lon_SST.copy(), lat_SST.copy(), sig_mask_18)

# SST shading
CS = ax_c.contourf(lon_1, lat_1, SST_1_plot_18, levels=np.arange(-6, 6.01, 0.25),
                   extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())
ax_c.contourf(lon_2, lat_2, SST_2_plot_18, levels=np.arange(-6, 6.01, 0.25),
              extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())
ax_c.contourf(lon_3, lat_3, SST_3_plot_18, levels=np.arange(-6, 6.01, 0.25),
              extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())
ax_c.contourf(lon_4, lat_4, SST_4_plot_18, levels=np.arange(-6, 6.01, 0.25),
              extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())

# overlay NON-significant regions with hatching
for lo, la, sig in [(lon_1, lat_1, sig1_18),
                    (lon_2, lat_2, sig2_18),
                    (lon_3, lat_3, sig3_18),
                    (lon_4, lat_4, sig4_18)]:
    ax_c.contourf(
        lo, la, sig,
        levels=[0.5, 1.5],
        hatches=['..'],
        colors='none',
        transform=ccrs.PlateCarree())

ax_c.set_global()
ax_c.coastlines(linewidth=0.6)
#ax_c.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.5)
ax_c.add_feature(cfeature.LAND, facecolor='tan', alpha=0.5)

#cbar2 = fig.colorbar(contourf2, ax=ax_c, orientation="horizontal", pad=0.05, fraction=0.05)
#cbar2.set_label('Temperature difference [$^\circ$C]', fontsize=11)
#cbar2.set_ticks([-9, -6, -3, 0, 3, 6, 9])

graph_1		= ax_d.plot(lat_atm, MHT_tot_off - MHT_tot_on, '-', color = 'k', linewidth = 2, label = 'Total')
graph_2		= ax_d.plot(lat_ocn, MHT_ocn_off - MHT_ocn_on, '-', color = 'royalblue', linewidth = 2, label = 'Ocean')
graph_3		= ax_d.plot(lat_atm, MHT_atm_off - MHT_atm_on, '-', color = 'firebrick', linewidth = 2, label = 'Atmosphere')


ax_d.set_ylabel('MHT difference (PW)', fontsize=11)
ax_d.set_xlim(-90, 90)
ax_d.set_ylim(-0.8, 0.8)
ax_d.grid()

ax_d.set_xticks(np.arange(-90, 90.1, 30))
ax_d.set_xticklabels(['90$^{\circ}$S', '60$^{\circ}$S', '30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])

graphs	      	= graph_1 + graph_2 + graph_3

legend_labels 	= [l.get_label() for l in graphs]
legend_2	= ax_d.legend(graphs, legend_labels, loc = 'upper left', ncol=1, framealpha = 1.0)


ax_d.set_title('d) Meridional heat transport (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)


plt.tight_layout()
plt.savefig(directory_figures +'Figure_1_CD.pdf')
plt.show()

#%%

fig = plt.figure(figsize=(17, 4))
gs = fig.add_gridspec(1, 3)
                            
ax_a = fig.add_subplot(gs[0])                            
ax_c = fig.add_subplot(gs[1], projection=ccrs.Robinson()) 
ax_d = fig.add_subplot(gs[2])                            

CS_black = ax_a.contour(lat_amoc, depth_amoc, AMOC_2, colors='black', levels=np.linspace(0, 23, 5))
ax_a.clabel(CS_black, inline=True, fontsize=10, fmt='%1.1f')  

CS1 = ax_a.contourf(lat_amoc, depth_amoc, AMOC_3 - AMOC_2, cmap='RdBu_r', levels=np.linspace(-13, 13, 21), extend='both')
cbar = fig.colorbar(CS1)
cbar.set_label('AMOC difference [Sv]',  fontsize=11)
ax_a.set_ylim(depth[-1], 0)
ax_a.set_title('a) AMOC streamfunction (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=13)
ax_a.set_xticks(np.arange(-30, 90.1, 30))
ax_a.set_xticklabels(['30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])
ax_a.set_ylabel('Depth [km]', fontsize=12)

ax_c.set_global()
ax_c.coastlines(linewidth=0.6)
#ax_c.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.5)
ax_c.add_feature(cfeature.LAND, facecolor='tan', alpha=0.5)

# =========================================================
CS      = ax_c.contourf(lon_1, lat_1, SST_1_plot_45, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax_c.contourf(lon_2, lat_2, SST_2_plot_45, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax_c.contourf(lon_3, lat_3, SST_3_plot_45, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())
CS      = ax_c.contourf(lon_4, lat_4, SST_4_plot_45, levels = np.arange(-6, 6.01, 0.25), extend = 'both', cmap = 'RdBu_r', transform=ccrs.PlateCarree())

cb = fig.colorbar(CS, ax=ax_c, orientation='horizontal', pad=0.05, fraction=0.05, shrink=1)
cb.set_label('SST difference [$^\\circ$C]', fontsize=11)
cb.set_ticks([-6, -4, -2, 0, 2, 4, 6])
ax_c.set_title(r'c) Annual SST (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=13)

# for lat_i in range(0, SST_2_annual.shape[1], 10):
#     for lon_i in range(0, SST_2_annual.shape[2], 10):
#         if SST_2_annual[0,lat_i, lon_i] is ma.masked:
#             #print('masked')
#             continue
#         p_value = Welch(SST_2_annual[:, lat_i, lon_i], SST_3_annual[:, lat_i, lon_i])
#         if p_value <= 0.95:
#             ax_c.scatter(
#                 lon_SST[lat_i, lon_i], lat_SST[lat_i, lon_i],
#                 marker='o', edgecolor='k', s=6, facecolors='none',
#                 transform=ccrs.PlateCarree())

sig_mask_45 = ma.masked_all((SST_2_annual.shape[1], SST_2_annual.shape[2]))

for lat_i in range(SST_2_annual.shape[1]):
    for lon_i in range(SST_2_annual.shape[2]):
        # skip masked ocean/land points
        if ma.getmaskarray(SST_2_annual[:, lat_i, lon_i]).all():
            continue
        if ma.getmaskarray(SST_3_annual[:, lat_i, lon_i]).all():
            continue

        p_value = Welch(SST_2_annual[:, lat_i, lon_i], SST_3_annual[:, lat_i, lon_i])

        # mark NON-significant points
        if p_value <= 0.95:
            sig_mask_45[lat_i, lon_i] = 1.0

_, _, sig1_45, _, _, sig2_45, _, _, sig3_45, _, _, sig4_45 = LowCESMPlot(
    lon_SST.copy(), lat_SST.copy(), sig_mask_45)

# SST shading
CS = ax_c.contourf(lon_1, lat_1, SST_1_plot_45, levels=np.arange(-6, 6.01, 0.25),
                   extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())
ax_c.contourf(lon_2, lat_2, SST_2_plot_45, levels=np.arange(-6, 6.01, 0.25),
              extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())
ax_c.contourf(lon_3, lat_3, SST_3_plot_45, levels=np.arange(-6, 6.01, 0.25),
              extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())
ax_c.contourf(lon_4, lat_4, SST_4_plot_45, levels=np.arange(-6, 6.01, 0.25),
              extend='both', cmap='RdBu_r', transform=ccrs.PlateCarree())

# overlay NON-significant regions with hatching
for lo, la, sig in [(lon_1, lat_1, sig1_45),
                    (lon_2, lat_2, sig2_45),
                    (lon_3, lat_3, sig3_45),
                    (lon_4, lat_4, sig4_45)]:
    ax_c.contourf(
        lo, la, sig,
        levels=[0.5, 1.5],
        hatches=['..'],
        colors='none',
        transform=ccrs.PlateCarree())

ax_c.set_global()
ax_c.coastlines(linewidth=0.6)
#ax_c.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.5)

graph_1		= ax_d.plot(lat_atm, MHT_tot_off_45 - MHT_tot_on_45, '-', color = 'k', linewidth = 2, label = 'Total')
graph_2		= ax_d.plot(lat_ocn, MHT_ocn_off_45 - MHT_ocn_on_45, '-', color = 'royalblue', linewidth = 2, label = 'Ocean')
graph_3		= ax_d.plot(lat_atm, MHT_atm_off_45 - MHT_atm_on_45, '-', color = 'firebrick', linewidth = 2, label = 'Atmosphere')


ax_d.set_ylabel('MHT difference (PW)', fontsize=11)
ax_d.set_xlim(-90, 90)
ax_d.set_ylim(-0.8, 0.8)
ax_d.grid()

ax_d.set_xticks(np.arange(-90, 90.1, 30))
ax_d.set_xticklabels(['90$^{\circ}$S', '60$^{\circ}$S', '30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])

graphs	      	= graph_1 + graph_2 + graph_3

legend_labels 	= [l.get_label() for l in graphs]
legend_2	= ax_d.legend(graphs, legend_labels, loc = 'upper left', ncol=1, framealpha = 1.0)


ax_d.set_title('c) Meridional heat transport (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=13)


plt.tight_layout()
plt.savefig(directory_figures +'Figure_S1_CD.pdf')
plt.show()

#%%

fig = plt.figure(figsize=(17, 4))
gs = fig.add_gridspec(1, 3)
                            
ax_a = fig.add_subplot(gs[0])                            
ax_c = fig.add_subplot(gs[1], projection=ccrs.Robinson()) 
ax_d = fig.add_subplot(gs[2])                            

CS_black = ax_a.contour(lat_amoc, depth_amoc, AMOC_1, colors='black', levels=np.linspace(0, 23, 5))
ax_a.clabel(CS_black, inline=True, fontsize=10, fmt='%1.1f')  

CS1 = ax_a.contourf(lat_amoc, depth_amoc, AMOC_4 - AMOC_1, cmap='RdBu_r', levels=np.linspace(-13, 13, 21), extend='both')
cbar = fig.colorbar(CS1)
cbar.set_label('AMOC difference [Sv]',  fontsize=11)
ax_a.set_ylim(depth[-1], 0)
ax_a.set_title('a) AMOC streamfunction (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)
ax_a.set_xticks(np.arange(-30, 90.1, 30))
ax_a.set_xticklabels(['30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])
ax_a.set_ylabel('Depth [km]', fontsize=12)

ax_c.set_global()
ax_c.coastlines(linewidth=0.6)
ax_c.add_feature(cfeature.BORDERS, linewidth=0.2, alpha=0.5)

contourf2 = ax_c.contourf(
    lon, lat,
    np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0),
    transform=ccrs.PlateCarree(),            
    levels=np.linspace(-10, 10, 41),
    cmap='RdBu_r',
    extend='both')

cbar2 = fig.colorbar(contourf2, ax=ax_c, orientation="horizontal", pad=0.05, fraction=0.05)
cbar2.set_label('Temperature difference [$^\circ$C]', fontsize=11)
cbar2.set_ticks([-9, -6, -3, 0, 3, 6, 9])

#axs[1,0].set_xticks(np.arange(-180,181, 60), crs=ccrs.PlateCarree())
#lon_formatter = cticker.LongitudeFormatter()
#axs[1,0].xaxis.set_major_formatter(lon_formatter)
#axs[1,0].set_yticks(np.arange(-90,91,30), crs=ccrs.PlateCarree())
#lat_formatter = cticker.LatitudeFormatter()
#axs[1,0].yaxis.set_major_formatter(lat_formatter)
#axs[1,0].set_ylim(-10, 10)
#axs[1].set_ylabel('Latitude [$^\circ$N]', fontsize=12)
#axs[1].set_xlabel('Longitude [$^\circ$E]', fontsize=12)
ax_c.set_title('b) Annual 2m temperature (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)

for lat_i in range(0, len(lat), 2):
    for lon_i in range(0, len(lon), 2):
        #Determine significant difference
        p_value = Welch(TEMP_1_annual[:, lat_i, lon_i], TEMP_4_annual[:, lat_i, lon_i])

        if p_value <= 0.95:
            #Non-significant difference
            ax_c.scatter(lon[lon_i], lat[lat_i], marker = 'o', edgecolor = 'k' , s = 6, facecolors='none')


graph_1		= ax_d.plot(lat_atm, MHT_tot_off - MHT_tot_on, '-', color = 'k', linewidth = 2, label = 'Total')
graph_2		= ax_d.plot(lat_ocn, MHT_ocn_off - MHT_ocn_on, '-', color = 'royalblue', linewidth = 2, label = 'Ocean')
graph_3		= ax_d.plot(lat_atm, MHT_atm_off - MHT_atm_on, '-', color = 'firebrick', linewidth = 2, label = 'Atmosphere')


ax_d.set_ylabel('MHT difference (PW)', fontsize=11)
ax_d.set_xlim(-90, 90)
ax_d.set_ylim(-1.2, 1.2)
ax_d.grid()

ax_d.set_xticks(np.arange(-90, 90.1, 30))
ax_d.set_xticklabels(['90$^{\circ}$S', '60$^{\circ}$S', '30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])

graphs	      	= graph_1 + graph_2 + graph_3

legend_labels 	= [l.get_label() for l in graphs]
legend_2	= ax_d.legend(graphs, legend_labels, loc = 'upper left', ncol=1, framealpha = 1.0)


ax_d.set_title('c) Meridional heat transport (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)


plt.tight_layout()
plt.savefig(directory_figures +'Figure_OS_meanstate_18.pdf')
plt.show()

#%% Fourier analysis AMOC on 

period_min = 5
label_level = 10e5

time_forward    = time_transient[600:1500]
time_backward   = time_transient[2900:3800]

AMOC_forward    = AMOC_transient[600:1500]
AMOC_backward   = AMOC_transient[2900:3800]

#Detrend the time series to remove the long-term trend (which can affect the power spectrum)
AMOC_forward_detrended = TrendRemover(time_forward, AMOC_forward, 2)
AMOC_backward_detrended = TrendRemover(time_backward, AMOC_backward, 2)

freq, power, cl90, cl95, cl99, a = power_spectrum_rednoise(AMOC_forward_detrended.copy(), time_forward)
freq_b, power_b, cl90_b, cl95_b, cl99_b, a = power_spectrum_rednoise(AMOC_backward_detrended.copy(), time_backward)

plt.figure()
plt.plot(freq, power, color='k', lw=1, label='AMOC')
plt.plot(freq, cl90, 'g', lw=1.5, label='90%')
plt.plot(freq, cl95, 'b', lw=1.5, label='95%')
plt.plot(freq, cl99, 'r', lw=1.5, label='99%')
plt.ylim(10**(1.0), 10**(7.0))

for freq_i in range(1, len(freq)):
        #print(freq_i)
        #indicate period of significance
    period	= int(round(1.0 / freq[freq_i], 0))
    #print(int(round(1.0 / freq[freq_i], 0)))
        
        #break

    if period < period_min:
        print(period)
        print(period_min)
        break
        
    if power[freq_i] > cl99[freq_i]:
		#99% confidence level
            color_sig	= 'r'

    elif power[freq_i] > cl95[freq_i]:
		#95% confidence level
        color_sig	= 'b'

    elif power[freq_i] > cl90[freq_i]:
		#90% confidence level
        color_sig	= 'g'
    
    if power[freq_i] > cl90[freq_i]:
    #Only plot above 90% confidence level
        print(power[freq_i])
        print(cl90[freq_i])
        plt.text(freq[freq_i], label_level, str(period), horizontalalignment='center', verticalalignment='bottom', color = color_sig, fontsize=11)
        plt.plot([freq[freq_i], freq[freq_i]], [power[freq_i], label_level], ':'+color_sig, linewidth = 1.5)



plt.xscale('log')
plt.yscale('log')
plt.title('Spectral power AMOC forward', fontsize=14)
plt.grid(True)

#%%
plt.figure()
plt.plot(freq_b, power_b, color='k', lw=1, label='AMOC')
plt.plot(freq_b, cl90_b, 'g', lw=1.5, label='90%')
plt.plot(freq_b, cl95_b, 'b', lw=1.5, label='95%')
plt.plot(freq_b, cl99_b, 'r', lw=1.5, label='99%')
plt.ylim(10**(1.0), 10**(7.0))

for freq_i in range(1, len(freq_b)):
        #print(freq_i)
        #indicate period of significance
    period	= int(round(1.0 / freq_b[freq_i], 0))
    #print(int(round(1.0 / freq[freq_i], 0)))
        
        #break

    if period < period_min:
        print(period)
        print(period_min)
        break
        
    if power_b[freq_i] > cl99_b[freq_i]:
		#99% confidence level
            color_sig	= 'r'

    elif power_b[freq_i] > cl95_b[freq_i]:
		#95% confidence level
        color_sig	= 'b'

    elif power_b[freq_i] > cl90_b[freq_i]:
		#90% confidence level
        color_sig	= 'g'
    
    if power_b[freq_i] > cl90_b[freq_i]:
    #Only plot above 90% confidence level
        print(power_b[freq_i])
        print(cl90_b[freq_i])
        plt.text(freq_b[freq_i], label_level, str(period), horizontalalignment='center', verticalalignment='bottom', color = color_sig, fontsize=11)
        plt.plot([freq_b[freq_i], freq_b[freq_i]], [power_b[freq_i], label_level], ':'+color_sig, linewidth = 1.5)

plt.xscale('log')
plt.yscale('log')
plt.title('Spectral power AMOC backward', fontsize=14)
plt.grid(True)

#%% Multitaper PSD of AMOC strength with AR(1) confidence intervals (not normalised, so absolute units)

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

def zscore_nan_1d(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 10:
        return x * np.nan
    mu = np.nanmean(x[m])
    sig = np.nanstd(x[m], ddof=1)
    return (x - mu) / sig

def ar1_phi(x):
    """Lag-1 autocorrelation estimate (clipped)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = (x - x.mean()) / x.std(ddof=1)
    phi = np.corrcoef(x[:-1], x[1:])[0, 1]
    return float(np.clip(phi, -0.99, 0.99))

def mtm_psd_ar1_ci(x, fs=1.0, NW=2.0, Kmax=None, nsurr=2000, ci=(90,95,99), seed=0):
    """
    MTM PSD of data + AR(1) surrogate percentile envelopes, computed using the same MTM settings.
    Returns f, S_data, ci_dict, phi
    """
    rng = np.random.default_rng(seed)

    xz = zscore_nan_1d(x)
    xz = xz[np.isfinite(xz)]

    # Data PSD
    f, S = mtm_psd(xz, fs=fs, NW=NW, Kmax=Kmax)

    # AR(1) parameters
    phi = ar1_phi(xz)
    var = np.var(xz)
    b = np.sqrt((1 - phi**2) * var)

    n = len(xz)
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

        # standardize surrogate
        y = (y - y.mean()) / y.std(ddof=1)

        _, Sy = mtm_psd(y, fs=fs, NW=NW, Kmax=Kmax)
        S_surr[i, :] = Sy

    ci_dict = {p: np.percentile(S_surr, p, axis=0) for p in ci}
    return f, S, ci_dict, phi


#Settings
fs = 1.0 #Yearly data, so sampling frequency is 1 per year
NW = 1.0 #Time-bandwidth product, common choice is 2.0 for moderate resolution and variance reduction
Kmax = 2 #Number of tapers, often set to int(2*NW) - 1, but can be adjusted. Here we use 4 tapers for better variance reduction at the cost of some resolution.
nsurr = 10000
ci_level = 95

# --- compute absolute PSDs + envelopes ---
f_amoc_forward,  S_amoc_forward,  ci_amoc_forward,  phi_amoc_forward        = mtm_psd_ar1_ci(AMOC_forward_detrended,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,))
f_amoc_backward,  S_amoc_backward,  ci_amoc_backward,  phi_amoc_backward    = mtm_psd_ar1_ci(AMOC_backward_detrended,  fs=fs, NW=NW, Kmax=Kmax, nsurr=nsurr, ci=(ci_level,))

def to_period_sorted(f, S, CI):
    m = f > 0
    per = 1.0 / f[m]
    Sp  = S[m]
    CIp = CI[m]
    srt = np.argsort(per)
    return per[srt], Sp[srt], CIp[srt]

per_amoc_forward,  Sp_amoc_forward,  CIp_amoc_forward  = to_period_sorted(f_amoc_forward,  S_amoc_forward,  ci_amoc_forward[ci_level])
per_amoc_backward,  Sp_amoc_backward,  CIp_amoc_backward  = to_period_sorted(f_amoc_backward,  S_amoc_backward,  ci_amoc_backward[ci_level])

# %%

period_xlim = (2, 50)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)

spectra_panels = [
    (axes[0], per_amoc_forward, Sp_amoc_forward, CIp_amoc_forward, r"a) PI$^{on}_{QE}$ - AMOC index", "royalblue", "MT spectrum PI$^{on}_{QE}$"),
    (axes[1], per_amoc_backward, Sp_amoc_backward, CIp_amoc_backward, r"b) PI$^{off}_{QE}$ - AMOC index", "royalblue", "MT spectrum PI$^{off}_{QE}$")]

for ax, per, Sp, CIp, title, col, label in spectra_panels:
    ax.plot(per, Sp, lw=1.8, color=col, label=label)
    ax.plot(per, CIp, lw=1.2, color="black", ls="--")
    ax.set_ylim(0, 0.01)
    ax.set_xlim(*period_xlim)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

axes[0].set_ylabel(r"Power [normalised]", fontsize=11)
axes[0].set_xlabel("Period [model years]", fontsize=11)
axes[1].set_xlabel("Period [model years]", fontsize=11)

axes[0].legend(frameon=False, loc="upper left")
axes[1].legend(frameon=False, loc="upper left")  

fig.tight_layout()
plt.savefig(directory_figures +'Figure_S2_AMOC_spectra.pdf')
plt.show()

#%%

fig, ax = plt.subplots(figsize=(8, 5))

ax3 	= fig.add_axes([0.097, 0.18, 0.32, 0.4], projection = ccrs.Orthographic(-30, 10))

ax3.coastlines(resolution='110m')
ax3.gridlines()
ax3.add_feature(cfeature.LAND, zorder=10)
ax3.set_global()


lon1     = np.arange(0, 361)
lat1     = np.arange(-90, 91)
field   = np.ones((len(lat1), len(lon1))) * -0.35
CS      = ax3.contourf(lon1, lat1, field, levels = np.arange(-1, 1.01, 0.05), extend = 'both', cmap = 'BrBG', transform=ccrs.PlateCarree())

lon2     = np.arange(-100, -5)
lat2     = np.arange(20, 43)
field   = np.ones((len(lat2), len(lon2))) * 0.35
CS      = ax3.contourf(lon2, lat2, field, levels = np.arange(-1, 1.01, 0.05), extend = 'both', cmap = 'BrBG', transform=ccrs.PlateCarree())

lon3     = np.arange(-100, 3)
lat3     = np.arange(42, 51)
field   = np.ones((len(lat3), len(lon3))) * 0.35
CS      = ax3.contourf(lon3, lat3, field, levels = np.arange(-1, 1.01, 0.05), extend = 'both', cmap = 'BrBG', transform=ccrs.PlateCarree())

ax3.text(320, 38, '$+F_H$', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=12, transform=ccrs.PlateCarree())
ax3.text(340, -10, '$-F_H$', verticalalignment='center', horizontalalignment='center', color = 'k', fontsize=12, transform=ccrs.PlateCarree())

x_1	= np.arange(-65, 7.1, 0.1)
y_1	= np.zeros(len(x_1)) + 60.0
y_2	= np.arange(58, 62.01, 0.1)
x_2	= np.zeros(len(y_2)) + x_1[0]
y_3	= np.arange(58, 62.01, 0.1)
x_3	= np.zeros(len(y_3)) + x_1[-1]

#ax3.plot(x_1, y_1, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)
#ax3.plot(x_2, y_2, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)
#ax3.plot(x_3, y_3, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)

x_1	= np.arange(-81, -9.99, 0.1)
y_1	= np.zeros(len(x_1)) + 26.0
y_2	= np.arange(24, 28.01, 0.1)
x_2	= np.zeros(len(y_2)) + x_1[0]
y_3	= np.arange(24, 28.01, 0.1)
x_3	= np.zeros(len(y_3)) + x_1[-1]

#ax3.plot(x_1, y_1, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)
#ax3.plot(x_2, y_2, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)
#ax3.plot(x_3, y_3, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)

x_1	= np.arange(-60, 20.01, 0.1)
y_1	= np.zeros(len(x_1)) - 34
y_2	= np.arange(-37, -30.99, 0.1)
x_2	= np.zeros(len(y_2)) + x_1[0]
y_3	= np.arange(-37, -30.99, 0.1)
x_3	= np.zeros(len(y_3)) + x_1[-1]

#ax3.plot(x_1, y_1, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)
#ax3.plot(x_2, y_2, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)
#ax3.plot(x_3, y_3, '-k', linewidth = 2.0, transform=ccrs.PlateCarree(), zorder = 10)

#ax.plot(time_transient*0.0003, AMOC_max_transient[:, lat_idx_rapid], color = 'black', label='Transient', linewidth=1)

#ax.plot(time_branch1[0:500]*0.0003, AMOC_max_branch1[0:500, lat_idx_rapid], color = 'blue', label='E1')
#plt.vlines(x = time_branch1[0]*0.0003, ymin=np.min(AMOC_max_branch1[350:500, lat_idx_rapid]), ymax= np.max(AMOC_max_branch1[350:500, lat_idx_rapid]), color='orchid', linewidth=2)
#plt.hlines(xmin = time_branch1[0]*0.0003 - 20*0.0003, xmax = time_branch1[0]*0.0003 + 20*0.0003, y=np.min(AMOC_max_branch1[350:500, lat_idx_rapid]), color='orchid', linewidth=2)
#plt.hlines(xmin = time_branch1[0]*0.0003 - 20*0.0003, xmax = time_branch1[0]*0.0003 + 20*0.0003, y=np.max(AMOC_max_branch1[350:500, lat_idx_rapid]), color='orchid', linewidth=2)

#ax.plot(time_branch2[0:500]*0.0003, AMOC_max_branch2[0:500, lat_idx_rapid], color = 'red', label='E2')
#plt.vlines(x = time_branch2[0]*0.0003, ymin=np.min(AMOC_max_branch2[350:500, lat_idx_rapid]), ymax= np.max(AMOC_max_branch2[350:500, lat_idx_rapid]), color='red', linewidth=2)
#plt.hlines(xmin = time_branch2[0]*0.0003 - 20*0.0003, xmax = time_branch2[0]*0.0003 + 20*0.0003, y=np.min(AMOC_max_branch2[350:500, lat_idx_rapid]), color='red', linewidth=2)
#plt.hlines(xmin = time_branch2[0]*0.0003 - 20*0.0003, xmax = time_branch2[0]*0.0003 + 20*0.0003, y=np.max(AMOC_max_branch2[350:500, lat_idx_rapid]), color='red', linewidth=2)

#plt.plot(time_branch3[0]*0.0003 - 1250*0.0003, np.mean(AMOC_max_branch3[:, lat_idx_rapid]), 'o', color = 'darkturquoise', label='E3', markersize=10)
#plt.vlines(x = time_branch3[0]*0.0003 - 1250*0.0003, ymin=np.min(AMOC_max_branch3[:, lat_idx_rapid]), ymax= np.max(AMOC_max_branch3[:, lat_idx_rapid]), color='darkturquoise', linewidth=2)
#plt.hlines(xmin = (time_branch3[0] - 1250 - 20)*0.0003, xmax = (time_branch3[0] - 1250 + 20)*0.0003, y=np.min(AMOC_max_branch3[:, lat_idx_rapid]), color='darkturquoise', linewidth=2)
#plt.hlines(xmin = (time_branch3[0] - 1250 - 20)*0.0003, xmax = (time_branch3[0] - 1250 + 20)*0.0003, y=np.max(AMOC_max_branch3[:, lat_idx_rapid]), color='darkturquoise', linewidth=2)

#ax.set_ylim(-1,21)
#ax.set_xlim(0,0.66)
#ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
#ax.tick_params(axis='both', which='major', labelsize=12)
#ax.set_xlabel('Freshwater flux forcing F$_H$ [Sv]', fontsize=14)
#ax.set_ylabel('Freshwater transport [Sv]', fontsize=14)
#ax.set_title('AMOC strength at 26$^\circ$N', fontsize=17)
#ax.grid()
#ax.legend(fontsize=12, loc = 1)
plt.tight_layout()
plt.savefig(directory_figures +'Hosing location.pdf')
plt.show()

#%%

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.Robinson())

ax.set_global()

# Ocean (soft blue)
ax.add_feature(cfeature.OCEAN, facecolor='#c6dbef')  

# Land (warm brownish)
ax.add_feature(cfeature.LAND, facecolor='#c2b280')  

# Coastlines
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black')

# Optional: remove frame
ax.spines['geo'].set_visible(False)

plt.tight_layout()
plt.savefig(directory_figures + 'Earth.pdf')
plt.show()


# %% Difference SSTs

diff_18 = np.mean(SST_4_annual, axis=0) - np.mean(SST_1_annual, axis=0)
diff_45 = np.mean(SST_3_annual, axis=0) - np.mean(SST_2_annual, axis=0) 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.contourf(diff_18, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r')
ax1.set_title('SST difference (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=12)
ax1.set_xlabel('Longitude [$^\circ$E]', fontsize=11)
ax1.set_ylabel('Latitude [$^\circ$N]', fontsize=11)
ax1.colorbar = plt.colorbar(ax1.contourf(diff_18, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r'), ax=ax1)
ax1.grid()

ax2.contourf(diff_45, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r')
ax2.set_title('SST difference (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=12)
ax2 .set_xlabel('Longitude [$^\circ$E]', fontsize=11)
ax2.colorbar = plt.colorbar(ax2.contourf(diff_45, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r'), ax=ax2)
ax2.grid()  

ax3.contourf(abs(diff_18) - abs(diff_45), levels=np.arange(-3, 3.01, 0.25), extend='both', cmap='RdBu_r')
ax3.set_title('SST difference difference (|PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$| - |PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$|)', fontsize=12)
ax3 .set_xlabel('Longitude [$^\circ$E]', fontsize=11)
ax3.colorbar = plt.colorbar(ax3.contourf(abs(diff_18) - abs(diff_45), levels=np.arange(-3, 3.01, 0.25), extend='both', cmap='RdBu_r'), ax=ax3)
ax3.grid()  

plt.tight_layout()
#plt.savefig(directory_figures + 'SST_differences.pdf')
plt.show()
# %%

diff_18 = np.mean(TEMP_4_annual, axis=0) - np.mean(TEMP_1_annual, axis=0)
diff_45 = np.mean(TEMP_3_annual, axis=0) - np.mean(TEMP_2_annual, axis=0) 
plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.contourf(lon, lat, diff_18, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r')
ax1.set_title('2m air temperature difference (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=12)
ax1.set_xlabel('Longitude [$^\circ$E]', fontsize=11)
ax1.set_ylabel('Latitude [$^\circ$N]', fontsize=11)
ax1.colorbar = plt.colorbar(ax1.contourf(lon, lat, diff_18, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r'), ax=ax1)
ax1.grid()

ax2.contourf(lon, lat, diff_45, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r')
ax2.set_title('2m air temperature difference (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=12)
ax2 .set_xlabel('Longitude [$^\circ$E]', fontsize=11)
ax2.colorbar = plt.colorbar(ax2.contourf(lon, lat, diff_45, levels=np.arange(-6, 6.01, 0.25), extend='both', cmap='RdBu_r'), ax=ax2)
ax2.grid()  

ax3.contourf(lon, lat, abs(diff_18) - abs(diff_45), levels=np.arange(-3, 3.01, 0.25), extend='both', cmap='RdBu_r')
ax3.set_title('2m air temperature difference difference (|PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$| - |PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$|)', fontsize=12)
ax3 .set_xlabel('Longitude [$^\circ$E]', fontsize=11)
ax3.colorbar = plt.colorbar(ax3.contourf(lon, lat, abs(diff_18) - abs(diff_45), levels=np.arange(-3, 3.01, 0.25), extend='both', cmap='RdBu_r'), ax=ax3)
ax3.grid()  

plt.tight_layout()
#plt.savefig(directory_figures + 'SST_differences.pdf')
plt.show()
# %% Determine average SSTs in tropical North Atantic (TNA) box

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
    lon_min, lon_max = -75.0, 10

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

lat_min = -30
lat_max = 30

mask2d = north_atlantic_mask(lat_SST, lon_SST, lat_min=lat_min, lat_max=lat_max)
SST_1_masked = ma.masked_where(~mask2d, np.mean(SST_1_annual, axis=0))
SST_2_masked = ma.masked_where(~mask2d, np.mean(SST_2_annual, axis=0))
SST_3_masked = ma.masked_where(~mask2d, np.mean(SST_3_annual, axis=0))
SST_4_masked = ma.masked_where(~mask2d, np.mean(SST_4_annual, axis=0))

plt.figure()
plt.contourf(lon_SST, lat_SST, SST_1_masked, levels=np.arange(20, 30.01, 0.5), extend='both', cmap='RdBu_r')
plt.colorbar(label='SST [°C]')
plt.title('Average SST in TNA box (PI$^{\mathrm{on}}_{18}$)', fontsize=12)
plt.xlabel('Longitude [$^\circ$E]', fontsize=11)
plt.ylabel('Latitude [$^\circ$N]', fontsize=11)
plt.grid()
plt.tight_layout()

plt.figure()
plt.contourf(lon_SST, lat_SST, SST_4_masked - SST_1_masked, levels=np.arange(-2, 2.01, 0.5), extend='both', cmap='RdBu_r')
plt.colorbar(label='SST [°C]')
plt.title('Average SST difference in TNA box (PI$_{18}$)', fontsize=12)
plt.xlabel('Longitude [$^\circ$E]', fontsize=11)
plt.ylabel('Latitude [$^\circ$N]', fontsize=11)
plt.grid()
plt.tight_layout()

plt.figure()
plt.contourf(lon_SST, lat_SST, SST_3_masked - SST_2_masked, levels=np.arange(-2, 2.01, 0.5), extend='both', cmap='RdBu_r')
plt.colorbar(label='SST [°C]')
plt.title('Average SST difference in TNA box (PI$_{45}$)', fontsize=12)
plt.xlabel('Longitude [$^\circ$E]', fontsize=11)
plt.ylabel('Latitude [$^\circ$N]', fontsize=11)
plt.grid()
plt.tight_layout()

#%%

print("Average SST in TNA box (PI$^{\mathrm{on}}_{18}$):", np.mean(SST_1_masked))
print("Average SST in TNA box (PI$^{\mathrm{off}}_{18}$):", np.mean(SST_4_masked))
print("Average SST in TNA box (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$):", np.mean(SST_4_masked - SST_1_masked))
print("Average SST in TNA box (PI$^{\mathrm{on}}_{45}$):", np.mean(SST_2_masked))
print("Average SST in TNA box (PI$^{\mathrm{off}}_{45}$):", np.mean(SST_3_masked))
print("Average SST in TNA box (PI$^{\mathrm{off}}   _{45}$ - PI$^{\mathrm{on}}_{45}$):", np.mean(SST_3_masked - SST_2_masked))

print('Difference in amount of cooling in TNA box', (np.mean(SST_4_masked - SST_1_masked)) - (np.mean(SST_3_masked - SST_2_masked)))

# %%
