#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:49:35 2026

@author: 6008399

Plot of EOF NAO with easterlies 

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

#Making pathway to folder with all data
#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

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
month_start    = 12
month_end      = 14

#DJF
lon, lat, eof_E1, time_E1, PC_E1, VAR_E1, EOF_E1		= ReadinData(directory_data + 'EOF_NAO_SLP_E1_month_12_14_detrend1_CESM_year_999_1099.nc')
lon, lat, eof_E2, time_E2, PC_E2, VAR_E2, EOF_E2		= ReadinData(directory_data + 'EOF_NAO_SLP_E2_month_12_14_detrend1_CESM_year_1899_1999.nc')
lon, lat, eof_E3, time_E3, PC_E3, VAR_E3, EOF_E3		= ReadinData(directory_data + 'EOF_NAO_SLP_E3_month_12_14_detrend1_CESM_year_3299_3399.nc')
lon, lat, eof_E4, time_E4, PC_E4, VAR_E4, EOF_E4		= ReadinData(directory_data + 'EOF_NAO_SLP_E4_month_12_14_detrend1_CESM_year_4199_4299.nc')

#%% Take first EOF for NAO

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

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('First EOF mean sea level pressure January', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, -EOF_SLP_E1 * std(PC_E1[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar  = fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
cbar.set_label('[hPa]')
axs[0,0].set_title('a) EOF1 January SLP - PI$^{on}_{18}$ (var.ex. = '+str(int(VAR_E1[0]*100))+'%)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[0,1].contourf(lon, lat, EOF_SLP_E4* std(PC_E4[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar = fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
cbar.set_label('[hPa]')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) EOF1 January SLP - PI$^{off}_{18}$ (var.ex. = '+str(int(VAR_E4[0]*100))+'%)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1,0].contourf(lon, lat, -EOF_SLP_E2 * std(PC_E2[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar = fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
cbar.set_label('[hPa]')
axs[1,0].set_title('c) EOF1 January SLP - PI$^{on}_{45}$ (var.ex. = '+str(int(VAR_E2[0]*100))+'%)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[1,1].contourf(lon, lat, EOF_SLP_E3 * std(PC_E3[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar = fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
cbar.set_label('[hPa]')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) EOF1 January SLP - PI$^{off}_{45}$ (var.ex. = '+str(int(VAR_E3[0]*100))+'%)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'Figure_NAO_EOF_SM.pdf')
plt.show() 

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('First EOF mean sea level pressure January', fontsize=15)
    
c1 = axs[0,0].contourf(lon, lat, EOF_E1[1,:,:] * std(PC_E1[1,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar  = fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
cbar.set_label('[hPa]')
axs[0,0].set_title('a) EOF2 January SLP - PI$^{on}_{18}$ (var.ex. = '+str(int(VAR_E1[1]*100))+'%)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[0,1].contourf(lon, lat, -EOF_E4[1,:,:] * std(PC_E4[1,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar = fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
cbar.set_label('[hPa]')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) EOF2 January SLP - PI$^{off}_{18}$ (var.ex. = '+str(int(VAR_E4[1]*100))+'%)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1,0].contourf(lon, lat, EOF_E2[1,:,:] * std(PC_E2[1,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar = fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
cbar.set_label('[hPa]')
axs[1,0].set_title('c) EOF2 January SLP - PI$^{on}_{45}$ (var.ex. = '+str(int(VAR_E2[1]*100))+'%)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[1,1].contourf(lon, lat, EOF_E3[1,:,:] * std(PC_E3[1,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-10,10,21), cmap='RdBu_r', extend='max')
cbar = fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
cbar.set_label('[hPa]')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) EOF2 January SLP - PI$^{off}_{45}$ (var.ex. = '+str(int(VAR_E3[1]*100))+'%)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_NAO_EOF_SM.pdf')
plt.show() 

#%%

month_start	= 12
month_end	= 14

#-----------------------------------------------------------------------------------------

fh = netcdf.Dataset(directory_data+'Jet_200hPa_Atlantic_month_12-14_branch3800_year_4199_4300.nc', 'r')

time_all	= fh.variables['time'][:]
lon_vel		    = fh.variables['lon'][:] 			
lat_vel		    = fh.variables['lat'][:] 			
u_vel_all	= fh.variables['U'][:]
u_vel_2_all	= fh.variables['UU'][:] 			
v_vel_all	= fh.variables['V'][:] 				
v_vel_2_all	= fh.variables['VV'][:]	

fh.close()

plt.figure()
plt.contourf(lon_vel, lat_vel, np.mean(u_vel_all, axis=0), levels = np.linspace(-20,20,21), cmap='RdBu_r', extend='both')

#-----------------------------------------------------------------------------------------

fh = netcdf.Dataset(directory_data+'Jet_200hPa_Atlantic_month_12-14_branch600_year_999_1100.nc', 'r')
	
u_vel_ref	= fh.variables['U'][:] 
u_vel_2_ref	= fh.variables['UU'][:] 			
v_vel_ref	= fh.variables['V'][:] 			
v_vel_2_ref	= fh.variables['VV'][:] 			

fh.close()

#%%-----------------------------------------------------------------------------------------

Uprime_600 = u_vel_ref - u_vel_ref.mean(axis=0)
Vprime_600 = v_vel_ref - v_vel_ref.mean(axis=0)

Uprime_3800 = u_vel_all - u_vel_all.mean(axis=0)
Vprime_3800 = v_vel_all - v_vel_all.mean(axis=0)

EKE_600 = 0.5*(Uprime_600**2 + Vprime_600**2)  # (time, lat, lon)
EKE_mean_600 = EKE_600.mean(axis=0)        # storm-track intensity map

EKE_3800 = 0.5*(Uprime_3800**2 + Vprime_3800**2)  # (time, lat, lon)
EKE_mean_3800 = EKE_3800.mean(axis=0)        # storm-track intensity map

#Get the wind speed mean and direction
vel_speed	  = np.mean(np.sqrt(u_vel_2_all + v_vel_2_all), axis = 0)
vel_speed_ref = np.mean(np.sqrt(u_vel_2_ref + v_vel_2_ref), axis = 0)

u_vel_all	= np.mean(u_vel_all, axis = 0)
v_vel_all	= np.mean(v_vel_all, axis = 0)
u_vel_ref	= np.mean(u_vel_ref, axis = 0)
v_vel_ref	= np.mean(v_vel_ref, axis = 0)

vel_speed_plot 	= vel_speed - vel_speed_ref
u_vel_plot 	= u_vel_all - u_vel_ref
v_vel_plot 	= v_vel_all - v_vel_ref

#%%-----------------------------------------------------------------------------------------

fh = netcdf.Dataset(directory_data+'Jet_200hPa_Atlantic_month_12-14_branch2900_year_3299_3400.nc', 'r')

time_all_2900	       = fh.variables['time'][:]
lon_vel		           = fh.variables['lon'][:] 			
lat_vel		           = fh.variables['lat'][:] 			
u_vel_all_2900	       = fh.variables['U'][:]
u_vel_2_all_2900	   = fh.variables['UU'][:] 			
v_vel_all_2900	       = fh.variables['V'][:] 				
v_vel_2_all_2900	   = fh.variables['VV'][:]	

fh.close()

#-----------------------------------------------------------------------------------------

fh = netcdf.Dataset(directory_data+'Jet_200hPa_Atlantic_month_12-14_branch1500_year_1899_2000.nc', 'r')
	
u_vel_ref_1500	     = fh.variables['U'][:] 
u_vel_2_ref_1500	 = fh.variables['UU'][:] 			
v_vel_ref_1500	     = fh.variables['V'][:] 			
v_vel_2_ref_1500	 = fh.variables['VV'][:] 			

fh.close()

#-----------------------------------------------------------------------------------------

jetlat_on  = jet_latitude(u_vel_ref_1500,  lat_vel, lon_vel, lon1=-80, lon2=10)
jetlat_off = jet_latitude(u_vel_all_2900, lat_vel, lon_vel, lon1=-80, lon2=10)

print("Mean jet latitude ON :", jetlat_on.mean())
print("Mean jet latitude OFF:", jetlat_off.mean())
print("Shift (OFF-ON) [deg]:", jetlat_off.mean() - jetlat_on.mean())

Uprime_1500 = u_vel_ref_1500 - u_vel_ref_1500.mean(axis=0)
Vprime_1500 = v_vel_ref_1500 - v_vel_ref_1500.mean(axis=0)

Uprime_2900 = u_vel_all_2900 - u_vel_all_2900.mean(axis=0)
Vprime_2900 = v_vel_all_2900 - v_vel_all_2900.mean(axis=0)

EKE_1500 = 0.5*(Uprime_1500**2 + Vprime_1500**2)  # (time, lat, lon)
EKE_mean_1500 = EKE_1500.mean(axis=0)        # storm-track intensity map

EKE_2900 = 0.5*(Uprime_2900**2 + Vprime_2900**2)  # (time, lat, lon)
EKE_mean_2900 = EKE_2900.mean(axis=0)        # storm-track intensity map

#Get the wind speed mean and direction
vel_speed_2900	  = np.mean(np.sqrt(u_vel_2_all_2900 + v_vel_2_all_2900), axis = 0)
vel_speed_ref_1500 = np.mean(np.sqrt(u_vel_2_ref_1500 + v_vel_2_ref_1500), axis = 0)

u_vel_all_2900	= np.mean(u_vel_all_2900, axis = 0)
v_vel_all_2900	= np.mean(v_vel_all_2900, axis = 0)
u_vel_ref_1500	= np.mean(u_vel_ref_1500, axis = 0)
v_vel_ref_1500	= np.mean(v_vel_ref_1500, axis = 0)

vel_speed_plot_2900_1500 	  = vel_speed_2900 - vel_speed_ref_1500
u_vel_plot_2900_1500 	      = u_vel_all_2900 - u_vel_ref_1500
v_vel_plot_2900_1500 	      = v_vel_all_2900 - v_vel_ref_1500

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

CS      = ax.contourf(lon_vel, lat_vel, vel_speed_plot, levels = np.arange(-10, 10.1, 1), extend = 'both', cmap = 'PuOr_r', transform=ccrs.PlateCarree())

divider = make_axes_locatable(ax)
ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)

cbar    = colorbar(CS, ticks = np.arange(-10, 10.01, 5), cax=ax_cb)
cbar.set_label('Wind speed difference (m s$^{-1}$)')

scale_arrow	= 4
Q = ax.quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot[::scale_arrow, ::scale_arrow], v_vel_plot[::scale_arrow, ::scale_arrow], scale = 100, transform=ccrs.PlateCarree())

#qk = ax.quiverkey(Q, 0.17, 0.10, 10, '10 m s$^{-1}$', labelpos = 'S', coordinates='figure')

#ax.plot([-45, -45], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')
#ax.plot([15, 15], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_extent([-90, 30, -0.001, 80.001], ccrs.PlateCarree())
ax.coastlines('110m')
ax.add_feature(cfeature.LAND, zorder=0)

ax.set_title('d) Difference 200 hPa velocities (January, $\overline{F_H}$ = 0.18Sv)')
plt.tight_layout()
plt.savefig(directory_figures +'Figure_4_CD_B.pdf')
show()

#-----------------------------------------------------------------------------------------

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

CS      = ax.contourf(lon_vel, lat_vel, vel_speed_plot_2900_1500, levels = np.arange(-10, 10.1, 1), extend = 'both', cmap = 'PuOr_r', transform=ccrs.PlateCarree())

divider = make_axes_locatable(ax)
ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
fig.add_axes(ax_cb)

cbar    = colorbar(CS, ticks = np.arange(-10, 10.01, 5), cax=ax_cb)
cbar.set_label('Wind speed difference (m s$^{-1}$)')

scale_arrow	= 4
Q = ax.quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], v_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], scale = 100, transform=ccrs.PlateCarree())

#qk = ax.quiverkey(Q, 0.17, 0.10, 10, '10 m s$^{-1}$', labelpos = 'S', coordinates='figure')

#ax.plot([-45, -45], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')
#ax.plot([15, 15], [-10, 85], '--', linewidth = 2.0, color = 'royalblue')

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_extent([-90, 30, -0.001, 80.001], ccrs.PlateCarree())
ax.coastlines('110m')
ax.add_feature(cfeature.LAND, zorder=0)

ax.set_title('d) Difference 200 hPa velocities (January, $\overline{F_H}$ = 0.45Sv)')
plt.tight_layout()
#plt.savefig(directory_figures +'Figure_4_CD_B.pdf')
show()

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0,0].contourf(lon, lat, EOF_SLP_E1*std(PC_E1[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-7,7,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[0,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-6, 7, 3)) 
cbar.set_label('[hPa]')
axs[0,0].set_title('a) EOF1 pattern - DJF SLP (PI$^{\mathrm{on}}_{18}$)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

levels = [-7, -4, 0, 1, 2]

#cs = axs[1,0].contour(    lon, lat, EOF_SLP_E2*std(PC_E2[0,:]),
#    levels=levels,
#    colors='dimgrey',
#    linewidths=0.8,
#    transform=ccrs.PlateCarree())

c2 = axs[0,1].contourf(lon_vel, lat_vel, vel_speed_ref, transform=ccrs.PlateCarree(), levels = np.arange(10, 45.1, 1), extend = 'both', cmap = 'Spectral_r')
cbar = fig.colorbar(c2, ax=axs[0,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) 200 hPa velocities - DJF (PI$^{\mathrm{on}}_{18}$)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
axs[0,1].set_xlim(-90, 40)
axs[0,1].set_ylim(20, 80)
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

#divider = make_axes_locatable(axs[0,1])
#ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
#fig.add_axes(ax_cb)

#cbar    = colorbar(c2, ticks = np.arange(-10, 10.01, 5))
cbar.set_label('Wind speed [m s$^{-1}$]')

scale_arrow	= 4
Q = axs[0,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_ref[::scale_arrow, ::scale_arrow], v_vel_ref[::scale_arrow, ::scale_arrow], scale = 500, transform=ccrs.PlateCarree())

axs[0,1].quiverkey(
    Q,
    X=0.1, Y=-.17,          # position (relative to axis)
    U=30.0,                  # reference value (1 m/s)
    label='30 m/s',
    labelpos='E', 
    fontproperties={'size': 11})

c1 = axs[1,0].contourf(lon, lat, -EOF_SLP_E4*std(PC_E4[0,:]) - EOF_SLP_E1*std(PC_E1[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-3,3,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[1,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-3, 4, 1)) 
cbar.set_label('[hPa]')
axs[1,0].set_title('c) Difference EOF1 pattern - DJF SLP (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
axs[1,0].set_xlim(-90, 40)
axs[1,0].set_ylim(20, 80)
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

levels = [-7, -4, 0, 1, 2]

#cs = axs[1,0].contour(
#    lon, lat, -EOF_SLP_E1*std(PC_E1[0,:]),
#    levels=levels,
#    colors='dimgrey',
#    linewidths=0.8,
#    transform=ccrs.PlateCarree())


c2 = axs[1,1].contourf(lon_vel, lat_vel, vel_speed_plot, transform=ccrs.PlateCarree(), levels = np.arange(-10, 10.1, 1), extend = 'both', cmap = 'PuOr_r')
cbar = fig.colorbar(c2, ax=axs[1,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Difference 200 hPa velocities - DJF (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
axs[1,1].set_xlim(-90, 40)
axs[1,1].set_ylim(20, 80)
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

#divider = make_axes_locatable(axs[0,1])
#ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
#fig.add_axes(ax_cb)

#cbar    = colorbar(c2, ticks = np.arange(-10, 10.01, 5))
cbar.set_label('Wind speed difference [m s$^{-1}$]')

scale_arrow	= 4
Q2 = axs[1,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot[::scale_arrow, ::scale_arrow], v_vel_plot[::scale_arrow, ::scale_arrow], scale = 100, transform=ccrs.PlateCarree())

axs[1,1].quiverkey(
    Q2,
    X=0.1, Y=-.17,          # position (relative to axis)
    U=10.0,                  # reference value (1 m/s)
    label='10 m/s',
    labelpos='E', 
    fontproperties={'size': 11})

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'Figure_3_CD.pdf')
plt.show()

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:
    ax.coastlines()

c1 = axs[0,0].contourf(lon, lat, EOF_SLP_E2*std(PC_E2[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-7,7,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[0,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-6, 7, 3)) 
cbar.set_label('[hPa]')
axs[0,0].set_title('a) EOF1 pattern - DJF SLP (PI$^{\mathrm{on}}_{45}$)')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

levels = [-7, -4, 0, 1, 2]

#cs = axs[1,0].contour(    lon, lat, EOF_SLP_E2*std(PC_E2[0,:]),
#    levels=levels,
#    colors='dimgrey',
#    linewidths=0.8,
#    transform=ccrs.PlateCarree())

c2 = axs[0,1].contourf(lon_vel, lat_vel, vel_speed_ref_1500, transform=ccrs.PlateCarree(), levels = np.arange(10, 45.1, 1), extend = 'both', cmap = 'Spectral_r')
cbar = fig.colorbar(c2, ax=axs[0,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) 200 hPa velocities - DJF (PI$^{\mathrm{on}}_{45}$)')
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
axs[0,1].set_xlim(-90, 40)
axs[0,1].set_ylim(20, 80)
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

#divider = make_axes_locatable(axs[0,1])
#ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
#fig.add_axes(ax_cb)

#cbar    = colorbar(c2, ticks = np.arange(-10, 10.01, 5))
cbar.set_label('Wind speed [m s$^{-1}$]')

scale_arrow	= 4
Q = axs[0,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_ref[::scale_arrow, ::scale_arrow], v_vel_ref[::scale_arrow, ::scale_arrow], scale = 500, transform=ccrs.PlateCarree())

axs[0,1].quiverkey(
    Q,
    X=0.1, Y=-.17,          # position (relative to axis)
    U=30.0,                  # reference value (1 m/s)
    label='30 m/s',
    labelpos='E', 
    fontproperties={'size': 11})

c1 = axs[1,0].contourf(lon, lat, -EOF_SLP_E3*std(PC_E3[0,:]) - EOF_SLP_E2*std(PC_E2[0,:]), transform=ccrs.PlateCarree(), levels = np.linspace(-3,3,21), cmap='RdBu_r', extend='both')
cbar = fig.colorbar(c1, ax=axs[1,0], orientation='vertical', shrink=0.7)
cbar.set_ticks(np.arange(-3, 4, 1)) 
cbar.set_label('[hPa]')
axs[1,0].set_title('c) Difference EOF1 pattern - DJF SLP (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
axs[1,0].set_xlim(-90, 40)
axs[1,0].set_ylim(20, 80)
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

levels = [-7, -4, 0, 1, 2]

#cs = axs[1,0].contour(
#    lon, lat, -EOF_SLP_E1*std(PC_E1[0,:]),
#    levels=levels,
#    colors='dimgrey',
#    linewidths=0.8,
#    transform=ccrs.PlateCarree())


c2 = axs[1,1].contourf(lon_vel, lat_vel, vel_speed_plot_2900_1500, transform=ccrs.PlateCarree(), levels = np.arange(-10, 10.1, 1), extend = 'both', cmap = 'PuOr_r')
cbar = fig.colorbar(c2, ax=axs[1,1], orientation='vertical', shrink=0.7)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) Difference 200 hPa velocities - DJF (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
axs[1,1].set_xlim(-90, 40)
axs[1,1].set_ylim(20, 80)
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,90,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

#divider = make_axes_locatable(axs[0,1])
#ax_cb   = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
#fig.add_axes(ax_cb)

#cbar    = colorbar(c2, ticks = np.arange(-10, 10.01, 5))
cbar.set_label('Wind speed difference [m s$^{-1}$]')

scale_arrow	= 4
Q2 = axs[1,1].quiver(lon_vel[::scale_arrow], lat_vel[::scale_arrow], u_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], v_vel_plot_2900_1500[::scale_arrow, ::scale_arrow], scale = 100, transform=ccrs.PlateCarree())

axs[1,1].quiverkey(
    Q2,
    X=0.1, Y=-.17,          # position (relative to axis)
    U=10.0,                  # reference value (1 m/s)
    label='10 m/s',
    labelpos='E', 
    fontproperties={'size': 11})

# Adjust the layout
plt.tight_layout()
plt.savefig(directory_figures +'Figure_3_CD_45_SM.pdf')
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

#%% Storm track intensity map

plt.figure()
plt.contourf(lon_vel, lat_vel, EKE_mean_600)

plt.figure()
plt.contourf(lon_vel, lat_vel, EKE_mean_3800)

fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for ax in axs.flat:  
    ax.coastlines()
    
#plt.suptitle('First EOF mean sea level pressure January', fontsize=15)
    
c1 = axs[0,0].contourf(lon_vel, lat_vel, EKE_mean_3800 - EKE_mean_600, transform=ccrs.PlateCarree(), levels = np.linspace(-15,15,21), cmap = 'RdBu_r', extend='both')
fig.colorbar(c1, ax=axs[0,0], orientation='vertical')
axs[0,0].set_title('a) difference EKE - PI$^{off}_{18}$ - PI$^{on}_{18}$')
axs[0,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,0].set_xlim(-90, 40)
axs[0,0].set_ylim(20, 80)
axs[0,0].xaxis.set_major_formatter(lon_formatter)
axs[0,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[0,1].contourf(lon_vel, lat_vel, EKE_mean_2900 - EKE_mean_1500, transform=ccrs.PlateCarree(), levels = np.linspace(-15,15,21), cmap = 'RdBu_r', extend='max')
fig.colorbar(c2, ax=axs[0,1], orientation='vertical')
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[0,1].set_title('b) difference EKE - PI$^{off}_{45}$ - PI$^{on}_{45}$')
axs[0,1].set_xlim(-90, 40)
axs[0,1].set_ylim(20, 80)
axs[0,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[0,1].xaxis.set_major_formatter(lon_formatter)
axs[0,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[0,1].yaxis.set_major_formatter(lat_formatter)

c1 = axs[1,0].contourf(lon_vel, lat_vel, EKE_mean_1500, transform=ccrs.PlateCarree(), levels = np.linspace(0,50,21), extend='max')
fig.colorbar(c1, ax=axs[1,0], orientation='vertical')
axs[1,0].set_title('c) EKE - PI$^{on}_{45}$')
axs[1,0].set_xlim(-90, 40)
axs[1,0].set_ylim(20, 80)
axs[1,0].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,0].xaxis.set_major_formatter(lon_formatter)
axs[1,0].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,0].yaxis.set_major_formatter(lat_formatter)

c2 = axs[1,1].contourf(lon_vel, lat_vel, EKE_mean_2900, transform=ccrs.PlateCarree(), levels = np.linspace(0,50,21), extend='max')
fig.colorbar(c2, ax=axs[1,1], orientation='vertical')
axs[1,1].set_xlim(-90, 40)
axs[1,1].set_ylim(20, 80)
#axs[1].set_title('b) Second EOF (var.ex. = '+str(int(VAR[1]))+'%)')
axs[1,1].set_title('d) EKE - PI$^{off}_{45}$')
axs[1,1].set_xticks(np.arange(-90,31,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
axs[1,1].xaxis.set_major_formatter(lon_formatter)
axs[1,1].set_yticks(np.arange(20,81,20), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
axs[1,1].yaxis.set_major_formatter(lat_formatter)

# Adjust the layout
plt.tight_layout()
#plt.savefig(directory_figures +'EOF_SLP_ATLANTIC_January_moving_average_'+str(moving_average)+'_CESM_branches.pdf')
plt.show() 

# %%
