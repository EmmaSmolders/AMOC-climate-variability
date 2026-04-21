#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 21:57:37 2025

@author: 6008399

SAM indices

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
from numpy.fft import fft, fftfreq
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import linregress

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'


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

#%% Read in data

#Choose which months you want (i.e. DJF or JJA)
month_start = 1
month_end   = 12

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch600_year_999-1100.nc', 'r')

time_E1     = fh.variables['time_month'][0:101*12] #Model years
lon         = fh.variables['lon'][:]  #Array of longitudes [degE]
lat         = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_E1      = fh.variables['SLP_month'][0:101*12]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch1500_year_1899-2000.nc', 'r')

time_E2     = fh.variables['time_month'][0:101*12] #Model years
lon         = fh.variables['lon'][:]  #Array of longitudes [degE]
lat         = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_E2      = fh.variables['SLP_month'][0:101*12]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch2900_year_2900-3500.nc', 'r')

time_E3     = fh.variables['time_month'][399*12:500*12] #Model years
lon         = fh.variables['lon'][:]  #Array of longitudes [degE]
lat         = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_E3      = fh.variables['SLP_month'][399*12:500*12]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

fh      = netcdf.Dataset(directory_data+'SLP_month_'+str(month_start)+'-'+str(month_end)+'_branch3800_year_4199-4300.nc', 'r')

time_E4     = fh.variables['time_month'][0:101*12] #Model years
lon         = fh.variables['lon'][:]  #Array of longitudes [degE]
lat         = fh.variables['lat'][:]  #Array of latitudes [degN]
SLP_E4      = fh.variables['SLP_month'][0*12:101*12]   #Sea level pressure (av\eraged over months) [hPa]

fh.close()

#Read in data
fh      = netcdf.Dataset(directory_data+'Atmosphere_DX_DY_AREA.nc', 'r')

dy      = fh.variables['DY'][:] #Grid spacing in y-direction
dx      = fh.variables['DX'][:] #Grid spacing in x-direction
area    = fh.variables['AREA'][:]

fh.close()

#%%

def butter_highpass(data, cutoff_period, dt=1.0, order=2):
    # dt in years if annual data, or months if monthly converted consistently
    fs = 1 / dt
    fc = 1 / cutoff_period
    wn = fc / (0.5 * fs)
    b, a = butter(order, wn, btype='highpass')
    return filtfilt(b, a, data)

def nearest_lat_index(lat, target):
    return np.argmin(np.abs(lat - target))

# Example: zonal mean SLP
slp_zm_E1 = np.mean(SLP_E1, axis=2)   # (time, lat)
slp_zm_E2 = np.mean(SLP_E2, axis=2)   
slp_zm_E3 = np.mean(SLP_E3, axis=2)   
slp_zm_E4 = np.mean(SLP_E4, axis=2)   

i40 = nearest_lat_index(lat, -40)
i65 = nearest_lat_index(lat, -65)

P40_E1 = slp_zm_E1[:, i40]
P65_E1 = slp_zm_E1[:, i65]

P40_E2 = slp_zm_E2[:, i40]
P65_E2 = slp_zm_E2[:, i65]

P40_E3 = slp_zm_E3[:, i40]
P65_E3 = slp_zm_E3[:, i65]

P40_E4 = slp_zm_E4[:, i40]
P65_E4 = slp_zm_E4[:, i65]

# high-pass filtered series for std
P40_hp_E1 = butter_highpass(P40_E1, cutoff_period=50, dt=1.0, order=2)
P65_hp_E1 = butter_highpass(P65_E1, cutoff_period=50, dt=1.0, order=2)

P40_hp_E2 = butter_highpass(P40_E2, cutoff_period=50, dt=1.0, order=2)
P65_hp_E2 = butter_highpass(P65_E2, cutoff_period=50, dt=1.0, order=2)

P40_hp_E3 = butter_highpass(P40_E3, cutoff_period=50, dt=1.0, order=2)
P65_hp_E3 = butter_highpass(P65_E3, cutoff_period=50, dt=1.0, order=2)

P40_hp_E4 = butter_highpass(P40_E4, cutoff_period=50, dt=1.0, order=2)
P65_hp_E4 = butter_highpass(P65_E4, cutoff_period=50, dt=1.0, order=2)

P40_star_E1 = (P40_E1 - np.mean(P40_E1)) / np.std(P40_hp_E1)
P65_star_E1 = (P65_E1 - np.mean(P65_E1)) / np.std(P65_hp_E1)

P40_star_E2 = (P40_E2 - np.mean(P40_E2)) / np.std(P40_hp_E2)
P65_star_E2 = (P65_E2 - np.mean(P65_E2)) / np.std(P65_hp_E2)

P40_star_E3 = (P40_E3 - np.mean(P40_E3)) / np.std(P40_hp_E3)
P65_star_E3 = (P65_E3 - np.mean(P65_E3)) / np.std(P65_hp_E3)

P40_star_E4 = (P40_E4 - np.mean(P40_E4)) / np.std(P40_hp_E4)
P65_star_E4 = (P65_E4 - np.mean(P65_E4)) / np.std(P65_hp_E4)

SAM_E1 = P40_star_E1 - P65_star_E1
SAM_E2 = P40_star_E2 - P65_star_E2
SAM_E3 = P40_star_E3 - P65_star_E3
SAM_E4 = P40_star_E4 - P65_star_E4

plt.figure()
plt.plot(SAM_E1)
#plt.plot(SAM_index_E1)

time = np.arange(len(SAM_E1))
res_E1 = linregress(time, SAM_E1)
res_E2 = linregress(time, SAM_E2)
res_E3 = linregress(time, SAM_E3)
res_E4 = linregress(time, SAM_E4)

print(f"SAM E1 statistics: slope={res_E1.slope:.4f}, stderr={res_E1.stderr:.4f}, pvalue={res_E1.pvalue:.4f}")
print(f"SAM E2 statistics: slope={res_E2.slope:.4f}, stderr={res_E2.stderr:.4f}, pvalue={res_E2.pvalue:.4f}")
print(f"SAM E3 statistics: slope={res_E3.slope:.4f}, stderr={res_E3.stderr:.4f}, pvalue={res_E3.pvalue:.4f}")
print(f"SAM E4 statistics: slope={res_E4.slope:.4f}, stderr={res_E4.stderr:.4f}, pvalue={res_E4.pvalue:.4f}")

#%%

#Coordinates of Lisbon and Reykjavik
lat_lisbon  = 40
lat_reyk    = 65

def TrendRemover(time, data, trend_type):
	"""Removes trend of choice"""
	
	rank = polyfit(time, data, trend_type)
	fitting = 0.0 
		
	for rank_i in range(len(rank)):
			
		fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))

	data -= fitting
	
	return data

lat_index_lisbon    = (np.abs(lat - lat_lisbon)).argmin()
lat_index_reyk      = (np.abs(lat - lat_reyk)).argmin()

SLP_lisbon_E1          = np.mean(SLP_E1[:, lat_index_lisbon, :], axis=1)
SLP_reyk_E1            = np.mean(SLP_E1[:, lat_index_reyk, :], axis=1)

SLP_lisbon_E2          = np.mean(SLP_E2[:, lat_index_lisbon, :], axis=1)
SLP_reyk_E2            = np.mean(SLP_E2[:, lat_index_reyk, :], axis=1)

SLP_lisbon_E3          = np.mean(SLP_E3[:, lat_index_lisbon, :], axis=1)
SLP_reyk_E3            = np.mean(SLP_E3[:, lat_index_reyk, :], axis=1)

SLP_lisbon_E4          = np.mean(SLP_E4[:, lat_index_lisbon, :], axis=1)
SLP_reyk_E4            = np.mean(SLP_E4[:, lat_index_reyk, :], axis=1)

print('Data is normalised\n')
SLP_lisbon_E1	= SLP_lisbon_E1 - np.mean(SLP_lisbon_E1, axis = 0)
SLP_lisbon_E1	= SLP_lisbon_E1 / np.std(SLP_lisbon_E1, axis = 0)
SLP_reyk_E1	= SLP_reyk_E1 - np.mean(SLP_reyk_E1, axis = 0)
SLP_reyk_E1	= SLP_reyk_E1 / np.std(SLP_reyk_E1, axis = 0)

SLP_lisbon_E2	= SLP_lisbon_E2 - np.mean(SLP_lisbon_E2, axis = 0)
SLP_lisbon_E2	= SLP_lisbon_E2 / np.std(SLP_lisbon_E2, axis = 0)
SLP_reyk_E2	= SLP_reyk_E2 - np.mean(SLP_reyk_E2, axis = 0)
SLP_reyk_E2	= SLP_reyk_E2 / np.std(SLP_reyk_E2, axis = 0)

SLP_lisbon_E3	= SLP_lisbon_E3 - np.mean(SLP_lisbon_E3, axis = 0)
SLP_lisbon_E3	= SLP_lisbon_E3 / np.std(SLP_lisbon_E3, axis = 0)
SLP_reyk_E3	= SLP_reyk_E3 - np.mean(SLP_reyk_E3, axis = 0)
SLP_reyk_E3	= SLP_reyk_E3 / np.std(SLP_reyk_E3, axis = 0)

SLP_lisbon_E4	= SLP_lisbon_E4 - np.mean(SLP_lisbon_E4, axis = 0)
SLP_lisbon_E4	= SLP_lisbon_E4 / np.std(SLP_lisbon_E4, axis = 0)
SLP_reyk_E4	= SLP_reyk_E4 - np.mean(SLP_reyk_E4, axis = 0)
SLP_reyk_E4	= SLP_reyk_E4 / np.std(SLP_reyk_E4, axis = 0)

#SAM index is difference between mean SLP at 40S and 65S (zonal average or nearest stations)
SAM_index_E1 = SLP_lisbon_E1 - SLP_reyk_E1
SAM_index_E2 = SLP_lisbon_E2 - SLP_reyk_E2
SAM_index_E3 = SLP_lisbon_E3 - SLP_reyk_E3
SAM_index_E4 = SLP_lisbon_E4 - SLP_reyk_E4

#%%
#Central moving average
def Moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

window = 20


#%%


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

lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory + 'EOF_SAM_SLP_E1_month_1_12_moving_average_0_CESM_branch_year_999_1099.nc')
lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory + 'EOF_SAM_SLP_E2_month_1_12_moving_average_0_CESM_branch_year_1899_1999.nc')
lon, lat, eof_E3, time_E3, PC_E3, VAR_E3, EOF_E3		= ReadinData(directory + 'EOF_SAM_SLP_E3_month_1_12_moving_average_0_CESM_branch_year_3299_3399.nc')
lon, lat, eof_E4, time_E4, PC_E4, VAR_E4, EOF_E4		= ReadinData(directory + 'EOF_SAM_SLP_E4_month_1_12_moving_average_0_CESM_branch_year_4199_4299.nc')

#%%
window = 20

SAM_index_PC1_E1 = PC_E1[0,:]/std(PC_E1[0])
SAM_index_PC1_E2 = PC_E2[0,:]/std(PC_E2[0])
SAM_index_PC1_E3 = PC_E3[0,:]/std(PC_E3[0])
SAM_index_PC1_E4 = PC_E4[0,:]/std(PC_E4[0])

#%%

fig, axs = plt.subplots(2, 2, figsize=(10, 6))  

plt.suptitle('SAM index (DJF)', fontsize=14)

axs[0,0].set_title('a) SAM station index ($F_H$ = 0.18Sv)')
#axs[0,0].plot(time_E1 - time_E1[0], -SAM_index_E1, color='red', alpha = 0.3)
axs[0,0].plot(time_E1[window//2 : -window//2 + 1] - time_E1[0], -Moving_average(SAM_index_E1, window), color='red', label='AMOC on')
#axs[0,0].plot(time_E4 - time_E4[0], -SAM_index_E4, color='blue', alpha = 0.3)
axs[0,0].plot(time_E4[window//2 : -window//2 + 1] - time_E4[0], -Moving_average(SAM_index_E4, window), color='blue', label='AMOC off')
axs[0,0].set_ylim(-3, 3)
axs[0,0].legend()

axs[0,1].set_title('b) SAM station index ($F_H$ = 0.45Sv)')
#axs[0,1].plot(time_E2 - time_E2[0], -SAM_index_E2, color='red', alpha = 0.3)
axs[0,1].plot(time_E2[window//2 : -window//2 + 1] - time_E2[0], -Moving_average(SAM_index_E2, window), color='red', label='AMOC on')
#axs[0,1].plot(time_E3 - time_E3[0], -SAM_index_E3, color='blue', alpha = 0.3)
axs[0,1].plot(time_E3[window//2 : -window//2 + 1] - time_E3[0], -Moving_average(SAM_index_E3, window), color='blue', label='AMOC off')
axs[0,1].set_ylim(-3, 3)
axs[0,1].legend()

axs[1,0].set_title('c) SAM PC1 index ($F_H$ = 0.18Sv)')
#axs[1,0].plot(time_E1 - time_E1[0], SAM_index_PC1_E1, color='red', alpha = 0.3)
axs[1,0].plot(time_E1[window//2 : -window//2 + 1] - time_E1[0], Moving_average(SAM_index_PC1_E1, window), color='red', label='AMOC on')
#axs[1,0].plot(time_E4 - time_E4[0], SAM_index_PC1_E4, color='blue', alpha = 0.3)
axs[1,0].plot(time_E4[window//2 : -window//2 + 1] - time_E4[0], Moving_average(SAM_index_PC1_E4, window), color='blue', label='AMOC off')
axs[1,0].set_ylim(-3, 3)
axs[1,0].legend()
axs[1,0].set_xlabel('Time [model year]')

axs[1,1].set_title('d) SAM PC1 index ($F_H$ = 0.45Sv)')
#axs[1,1].plot(time_E2 - time_E2[0], SAM_index_PC1_E2, color='red', alpha = 0.3)
axs[1,1].plot(time_E2[window//2 : -window//2 + 1] - time_E2[0], Moving_average(SAM_index_PC1_E2, window), color='red', label='AMOC on')
#axs[1,1].plot(time_E3 - time_E3[0], SAM_index_PC1_E3, color='blue', alpha = 0.3)
axs[1,1].plot(time_E3[window//2 : -window//2 + 1] - time_E3[0], Moving_average(SAM_index_PC1_E3, window), color='blue', label='AMOC off')
axs[1,1].set_ylim(-3, 3)
axs[1,1].legend()
axs[1,1].set_xlabel('Time [model year]')

plt.tight_layout()
plt.savefig(directory_figures + 'SAM_index_month_'+str(month_start)+'_'+str(month_end)+'_branches.pdf')

#%%

plt.figure()
plt.plot(-Moving_average(SAM_index_E3, window))
plt.plot(Moving_average(SAM_index_PC1_E3, window))

#%%


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(SAM_index_PC1_E1, alpha=0.5, bins=40, color='red', label='AMOC on')
axes[0].hist(SAM_index_PC1_E4, alpha=0.5, bins=40, color='blue', label='AMOC off')
axes[0].legend()
axes[0].grid()
axes[0].set_title('a) SAM PC1 index ($F_H$=0.18Sv)')

axes[1].hist(SAM_E1, alpha=0.5, bins=40, color='red', label='AMOC on')
axes[1].hist(SAM_E4, alpha=0.5, bins=40, color='blue', label='AMOC off')
axes[1].legend()
axes[1].grid()
axes[1].set_title('b) SAM station index ($F_H$=0.18Sv)')

plt.savefig(directory_figures + 'PDF_SAM_index_Fh_018.pdf')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(SAM_index_PC1_E2, alpha=0.5, bins=40, color='red', label='AMOC on')
axes[0].hist(SAM_index_PC1_E3, alpha=0.5, bins=40, color='blue', label='AMOC off')
axes[0].legend()
axes[0].grid()
axes[0].set_title('c) SAM PC1 index ($F_H$=0.45Sv)')

axes[1].hist(SAM_E2, alpha=0.5, bins=40, color='red', label='AMOC on')
axes[1].hist(SAM_E3, alpha=0.5, bins=40, color='blue', label='AMOC off')
axes[1].legend()
axes[1].grid()
axes[1].set_title('d) SAM station index ($F_H$=0.45Sv)')

plt.savefig(directory_figures + 'PDF_SAM_index_Fh_045.pdf')
#%% Power spectra

#Forward
freq_f, power_f, cl90_f, cl95_f, cl99_f, a_f = power_spectrum_rednoise(
    SAM_index_PC1_E1,
    time_E1
)

#Backward
freq_b, power_b, cl90_b, cl95_b, cl99_b, a_b = power_spectrum_rednoise(
    SAM_index_PC1_E4,
    time_E4
)

freq_f_station, power_f_station, cl90_f_station, cl95_f_station, cl99_f_station, a_f = power_spectrum_rednoise(
    SAM_E1,
    time_E1
)

#Backward
freq_b_station, power_b_station, cl90_b_station, cl95_b_station, cl99_b_station, a_b = power_spectrum_rednoise(
    SAM_E4,
    time_E4
)

#%%

label_level		= 1.3 * 10**5.0	#Height of the labels for significant periods in plot
period_min		= 0			#Cut-off period (years) in plot

fig, axes = plt.subplots(
    1, 2, figsize=(10, 4),
    sharex=True, sharey=True
)

titles = ['a) SAM PC1 index (AMOC on, $F_H$ = 0.18Sv)', 'b) SAM PC1 index (AMOC off, $F_H$ = 0.18Sv))']
spectra = [
    (freq_f, power_f, cl90_f, cl95_f, cl99_f),
    (freq_b, power_b, cl90_b, cl95_b, cl99_b)
]

for ax, title, spec in zip(axes, titles, spectra):
    freq, power, cl90, cl95, cl99 = spec

    ax.plot(freq, power, color='k', lw=1, label='AMOC')
    #ax.plot(freq, cl90, 'g', lw=1.5, label='90%')
    ax.plot(freq, cl95, 'b', lw=1.5, label='95%')
    ax.plot(freq, cl99, 'r', lw=1.5, label='99%')
    
    ax.set_ylim(10**(1.0), 2*10**(5.0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    
    for freq_i in range(1, len(freq)):
        #print(freq_i)
        #indicate period of significance
        period	= int(round(1.0 / freq[freq_i], 0))
        #print(int(round(1.0 / freq[freq_i], 0)))
        
        #break

        if period < period_min:
        #print(period)
        #print(period_min)
            break
        
        if power[freq_i] > cl99[freq_i]:
		#99% confidence level
            color_sig	= 'r'

        elif power[freq_i] > cl95[freq_i]:
		#95% confidence level
            color_sig	= 'b'

	#elif power[freq_i] > cl90[freq_i]:
		#90% confidence level
	#	color_sig	= 'g'
    
        if power[freq_i] > cl95[freq_i]:
        #Only plot above 90% confidence level
            #print(power[freq_i])
            #print(cl99[freq_i])
            ax.text(freq[freq_i], label_level, str(period), horizontalalignment='center', verticalalignment='bottom', color = color_sig, fontsize=11)
            ax.plot([freq[freq_i], freq[freq_i]], [power[freq_i], label_level], ':'+color_sig, linewidth = 1.5)


axes[0].set_ylabel('Power', fontsize = 12)
axes[0].set_xlabel('Frequency [year$^{-1}$]', fontsize=12)
axes[1].set_xlabel('Frequency [year$^{-1}$]', fontsize=12)
#axes[1].legend(loc='lower left')

#plt.suptitle('AMOC Power Spectra (Red-noise significance)')
plt.tight_layout()
plt.savefig(directory_figures + 'Power_spectra_SAM_Fh_018_PC1.pdf')
plt.show()

#%%

fig, axes = plt.subplots(
    1, 2, figsize=(10, 4),
    sharex=True, sharey=True
)

titles = ['c) SAM station index (AMOC on, $F_H$ = 0.45Sv)', 'd) SAM station index (AMOC off, $F_H$ = 0.45Sv))']
spectra = [
    (freq_f_station, power_f_station, cl90_f_station, cl95_f_station, cl99_f_station),
    (freq_b_station, power_b_station, cl90_b_station, cl95_b_station, cl99_b_station)
]

for ax, title, spec in zip(axes, titles, spectra):
    freq, power, cl90, cl95, cl99 = spec

    ax.plot(freq, power, color='k', lw=1, label='AMOC')
    #ax.plot(freq, cl90, 'g', lw=1.5, label='90%')
    ax.plot(freq, cl95, 'b', lw=1.5, label='95%')
    ax.plot(freq, cl99, 'r', lw=1.5, label='99%')
    
    ax.set_ylim(10**(1.0), 2*10**(5.0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    
    for freq_i in range(1, len(freq)):
        #print(freq_i)
        #indicate period of significance
        period	= int(round(1.0 / freq[freq_i], 0))
        #print(int(round(1.0 / freq[freq_i], 0)))
        
        #break

        if period < period_min:
        #print(period)
        #print(period_min)
            break
        
        if power[freq_i] > cl99[freq_i]:
		#99% confidence level
            color_sig	= 'r'

        elif power[freq_i] > cl95[freq_i]:
		#95% confidence level
            color_sig	= 'b'

	#elif power[freq_i] > cl90[freq_i]:
		#90% confidence level
	#	color_sig	= 'g'
    
        if power[freq_i] > cl95[freq_i]:
        #Only plot above 90% confidence level
            #print(power[freq_i])
            #print(cl99[freq_i])
            ax.text(freq[freq_i], label_level, str(period), horizontalalignment='center', verticalalignment='bottom', color = color_sig, fontsize=11)
            ax.plot([freq[freq_i], freq[freq_i]], [power[freq_i], label_level], ':'+color_sig, linewidth = 1.5)


axes[0].set_ylabel('Power', fontsize = 12)
axes[0].set_xlabel('Frequency [year$^{-1}$]', fontsize=12)
axes[1].set_xlabel('Frequency [year$^{-1}$]', fontsize=12)
#axes[1].legend(loc='lower left')

#plt.suptitle('AMOC Power Spectra (Red-noise significance)')
plt.tight_layout()
plt.savefig(directory_figures + 'Power_spectra_SAM_Fh_045_station.pdf')
plt.show()

# %%
