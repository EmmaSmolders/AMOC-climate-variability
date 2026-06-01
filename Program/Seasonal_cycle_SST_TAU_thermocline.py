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

#Between 5S and 5N
lat1, lat2 = 167, 207

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

month_start = 1
month_end = 12

#not monthly data unfortuantlye..
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

print(np.min(lat), np.max(lat))

#%%

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

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Helper function
# =========================================================
def annual_removed_zonal(clim):
    """
    clim shape assumed: (12, lat, lon)
    returns:
        clim_anom   : annual-mean removed climatology (12, lat, lon)
        clim_zonal  : zonal/meridional averaged version for hovmoller-like plot
    """
    annual_mean = np.mean(clim, axis=0)          # (lat, lon)
    clim_anom = clim - annual_mean               # (12, lat, lon)
    clim_zonal = np.mean(clim_anom, axis=1)      # average over latitude -> (12, lon)
    return clim_anom, clim_zonal

# =========================================================
# Compute annual-removed SST climatologies
# =========================================================
clim_anom_E1, clim_zonal_E1 = annual_removed_zonal(clim_E1)
clim_anom_E2, clim_zonal_E2 = annual_removed_zonal(clim_E2)
clim_anom_E3, clim_zonal_E3 = annual_removed_zonal(clim_E3)
clim_anom_E4, clim_zonal_E4 = annual_removed_zonal(clim_E4)

# =========================================================
# Compute annual-removed TAUX climatologies
# Replace clim_taux_* with your actual TAUX climatology names
# =========================================================
clim_anom_taux_E1, clim_zonal_taux_E1 = annual_removed_zonal(clim_E1_taux)
clim_anom_taux_E2, clim_zonal_taux_E2 = annual_removed_zonal(clim_E2_taux)
clim_anom_taux_E3, clim_zonal_taux_E3 = annual_removed_zonal(clim_E3_taux)
clim_anom_taux_E4, clim_zonal_taux_E4 = annual_removed_zonal(clim_E4_taux)

months = np.arange(1, 13)

# If lon is 2D, use lon[0]; otherwise use lon directly
lon_plot = lon[0] if np.ndim(lon) > 1 else lon

# =========================================================
# 1) Annual-removed climatology plots for SST and TAUX
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

# ---- SST: PIoff18 annual-removed climatology
cf = axs[0, 0].contourf(
    lon_plot, months, clim_zonal_E1,
    levels=np.linspace(-2, 2, 21),
    extend='both', cmap='RdBu_r'
)
axs[0, 0].set_xlim(160, 280)
axs[0, 0].set_ylabel('Month')
axs[0, 0].set_title(r'a) Annual-removed climatological SST (5S–5N), PI$^{on}_{18}$')
cb = fig.colorbar(cf, ax=axs[0, 0])
cb.set_label('SST (°C)')

# ---- SST difference E4-E1
cf = axs[0, 1].contourf(
    lon_plot, months, clim_zonal_E4 - clim_zonal_E1,
    levels=np.linspace(-1, 1, 21),
    extend='both', cmap='RdBu_r'
)
axs[0, 1].set_xlim(160, 280)
axs[0, 1].set_ylabel('Month')
axs[0, 1].set_title(r'b) SST difference (PI$^{off}_{18}$ - PI$^{on}_{18}$)')
cb = fig.colorbar(cf, ax=axs[0, 1])
cb.set_label('SST (°C)')

# ---- TAUX: PIon18 annual-removed climatology
cf = axs[1, 0].contourf(
    lon_plot, months, clim_zonal_taux_E1,
    levels=np.linspace(-0.3, 0.3, 21),
    extend='both', cmap='RdBu_r'
)
axs[1, 0].set_xlim(160, 280)
axs[1, 0].set_ylabel('Month')
axs[1, 0].set_xlabel('Longitude')
axs[1, 0].set_title(r'c) Annual-removed climatological TAUX (5S–5N), PI$^{on}_{18}$')
cb = fig.colorbar(cf, ax=axs[1, 0])
cb.set_label('TAUX (N m$^{-2}$)')

# ---- TAUX difference E4-E1
cf = axs[1, 1].contourf(
    lon_plot, months, clim_zonal_taux_E4 - clim_zonal_taux_E1,
    levels=np.linspace(-0.1, 0.1, 21),
    extend='both', cmap='RdBu_r'
)
axs[1, 1].set_xlim(160, 280)
axs[1, 1].set_ylabel('Month')
axs[1, 1].set_xlabel('Longitude')
axs[1, 1].set_title(r'd) TAUX difference (PI$^{off}_{18}$ - PI$^{on}_{18}$)')
cb = fig.colorbar(cf, ax=axs[1, 1])
cb.set_label('TAUX (N m$^{-2}$)')

for ax in axs.flat:
    ax.set_xticks([160, 180, 200, 220, 240, 260, 280])
    ax.set_xticklabels(['160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W'])
    ax.set_yticks(months)

plt.show()

#%% =========================================================
# 2) 2x2 figure:
# filled contours = OFF - ON difference
# black contours = ON-state climatology
# Panels:
#   a) SST E4-E1 with E1 overlay
#   c) TAUX E4-E1 with E1 overlay
#   b) SST E3-E2 with E2 overlay
#   d) TAUX E3-E2 with E2 overlay
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 6))#, constrained_layout=True)

# Levels
sst_diff_levels = np.linspace(-1, 1, 21)
sst_cont_levels = np.linspace(-2, 2, 9)

taux_diff_levels = np.linspace(-0.1, 0.1, 21)
taux_cont_levels = np.linspace(-0.3, 0.3, 9)

# ---- a) SST E4-E1 with E1 overlay
cf = axs[0, 0].contourf(
    lon_plot, months, clim_zonal_E4 - clim_zonal_E1,
    levels=sst_diff_levels, extend='both', cmap='RdBu_r'
)
cs = axs[0, 0].contour(
    lon_plot, months, clim_zonal_E1,
    levels=sst_cont_levels, colors='k', linewidths=0.8
)
axs[0, 0].clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
axs[0, 0].set_title(r'a) SST (annual mean removed) (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)
axs[0, 0].set_ylabel('Month', fontsize=12)
cb = fig.colorbar(cf, ax=axs[0, 0])
cb.set_label('SST difference (°C)', fontsize=11)

# ---- b) SST E3-E2 with E2 overlay
cf = axs[0, 1].contourf(
    lon_plot, months, clim_zonal_E3 - clim_zonal_E2,
    levels=sst_diff_levels, extend='both', cmap='RdBu_r'
)
cs = axs[0, 1].contour(
    lon_plot, months, clim_zonal_E2,
    levels=sst_cont_levels, colors='k', linewidths=0.8
)
axs[0, 1].clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
axs[0, 1].set_title(r'b) SST (annual mean removed) (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=13)
axs[0, 1].set_ylabel('Month', fontsize=12)
#axs[1, 0].set_xlabel('Longitude')
cb = fig.colorbar(cf, ax=axs[0, 1])
cb.set_label('SST difference (°C)', fontsize=11)

# ---- c) TAUX E4-E1 with E1 overlay
cf = axs[1, 0].contourf(
    lon_plot, months, clim_zonal_taux_E4 - clim_zonal_taux_E1,
    levels=taux_diff_levels, extend='both', cmap='RdBu_r'
)
cs = axs[1, 0].contour(
    lon_plot, months, clim_zonal_taux_E1,
    levels=taux_cont_levels, colors='k', linewidths=0.8
)
axs[1, 0].clabel(cs, inline=True, fontsize=8, fmt='%1.3f')
axs[1, 0].set_title(r'c) Zonal wind stress (annual mean removed) (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)
axs[1, 0].set_ylabel('Month', fontsize=12)
cb = fig.colorbar(cf, ax=axs[1, 0])
cb.set_label('TAUX difference (N m$^{-2}$)', fontsize=11)

# ---- d) TAUX E3-E2 with E2 overlay
cf = axs[1, 1].contourf(
    lon_plot, months, clim_zonal_taux_E3 - clim_zonal_taux_E2,
    levels=taux_diff_levels, extend='both', cmap='RdBu_r'
)
cs = axs[1, 1].contour(
    lon_plot, months, clim_zonal_taux_E2,
    levels=taux_cont_levels, colors='k', linewidths=0.8
)
axs[1, 1].clabel(cs, inline=True, fontsize=8, fmt='%1.3f')
axs[1, 1].set_title(r'd) Zonal wind stress (annual mean removed) (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=13)
axs[1, 1].set_ylabel('Month', fontsize=12)
#axs[1, 1].set_xlabel('Longitude')
cb = fig.colorbar(cf, ax=axs[1, 1])
cb.set_label('TAUX difference (N m$^{-2}$)', fontsize=11)

for ax in axs.flat:
    ax.set_xlim(160, 280)
    ax.set_xticks([160, 180, 200, 220, 240, 260, 280])
    ax.set_xticklabels(['160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W'], fontsize=10)
    ax.set_yticks(months)

plt.tight_layout()
plt.savefig(directory_figures + 'Seasonal_cycle_SST_TAUX_annual_mean_removed.pdf', dpi=300)
plt.show()

#%%

fig, axs = plt.subplots(2, 2, figsize=(12, 6))#, constrained_layout=True)

# Levels
sst_diff_levels = np.linspace(-1, 1, 21)
sst_cont_levels = np.linspace(-2, 2, 9)

# ---- a) SST E1 
cf = axs[0, 0].contourf(
    lon_plot, months, clim_zonal_E1,
    levels=sst_diff_levels, extend='both', cmap='RdBu_r'
)

axs[0, 0].clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
axs[0, 0].set_title(r'a) SST (annual mean removed) (PI$^{\mathrm{on}}_{18}$)', fontsize=13)
axs[0, 0].set_ylabel('Month', fontsize=12)
cb = fig.colorbar(cf, ax=axs[0, 0])
cb.set_label('SST (°C)', fontsize=11)

# ---- a) SST E4 
cf = axs[0, 1].contourf(
    lon_plot, months, clim_zonal_E2,
    levels=sst_diff_levels, extend='both', cmap='RdBu_r'
)

axs[0, 1].clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
axs[0, 1].set_title(r'b) SST (annual mean removed) (PI$^{\mathrm{on}}_{45}$)', fontsize=13)
axs[0, 1].set_ylabel('Month', fontsize=12)
cb = fig.colorbar(cf, ax=axs[0, 1])
cb.set_label('SST (°C)', fontsize=11)

# ---- c) SST E4-E1 with E1 overlay
cf = axs[1, 0].contourf(
    lon_plot, months, clim_zonal_E4 - clim_zonal_E1,
    levels=sst_diff_levels, extend='both', cmap='RdBu_r'
)
cs = axs[1, 0].contour(
    lon_plot, months, clim_zonal_E1,
    levels=sst_cont_levels, colors='k', linewidths=0.8
)
axs[1, 0].clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
axs[1, 0].set_title(r'c) SST difference (PI$^{\mathrm{off}}_{18}$ - PI$^{\mathrm{on}}_{18}$)', fontsize=13)
axs[1, 0].set_ylabel('Month', fontsize=12)
cb = fig.colorbar(cf, ax=axs[1, 0])
cb.set_label('SST difference (°C)', fontsize=11)

# ---- d) SST E3-E2 with E2 overlay
cf = axs[1, 1].contourf(
    lon_plot, months, clim_zonal_E3 - clim_zonal_E2,
    levels=sst_diff_levels, extend='both', cmap='RdBu_r'
)
cs = axs[1, 1].contour(
    lon_plot, months, clim_zonal_E2,
    levels=sst_cont_levels, colors='k', linewidths=0.8
)
axs[1, 1].clabel(cs, inline=True, fontsize=8, fmt='%1.1f')
axs[1, 1].set_title(r'd) SST difference (PI$^{\mathrm{off}}_{45}$ - PI$^{\mathrm{on}}_{45}$)', fontsize=13)
axs[1, 1].set_ylabel('Month', fontsize=12)
#axs[1, 0].set_xlabel('Longitude')
cb = fig.colorbar(cf, ax=axs[1, 1])
cb.set_label('SST difference (°C)', fontsize=11)

for ax in axs.flat:
    ax.set_xlim(160, 280)
    ax.set_xticks([160, 180, 200, 220, 240, 260, 280])
    ax.set_xticklabels(['160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W'], fontsize=10)
    ax.set_yticks(months)

plt.tight_layout()
plt.savefig(directory_figures + 'Seasonal_cycle_SST_annual_mean_removed.pdf', dpi=300)
plt.show()

#%%


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
axs[0].set_title('a) (annual removed) Climatological monthly mean SST (5S-5N) for PI$^{off}_{18}$)')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), clim_zonal_E4 - clim_zonal_E1, levels=np.linspace(-1,1,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('SST (°C)')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Difference (PI$^{off}_{18}$ - PI$^{on}_{18}$)')

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
axs[0].set_title('a) (annual removed) Climatological monthly mean SST (5S-5N) for PI$^{on}_{45}$')

cf = axs[1].contourf(lon[0], np.linspace(1,12, 12), clim_zonal_E3 - clim_zonal_E2, levels=np.linspace(-1,1,21), extend='both', cmap='RdBu_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs[1].set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs[1].set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs[1])
colorbar.set_label('SST (°C)')
axs[1].set_ylabel('Month')
axs[1].set_title('b) Difference (PI$^{off}_{45}$ - PI$^{on}_{45}$)')



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
cf = axs.contourf(lon[0], np.linspace(1,12, 12), np.mean(clim_E1, axis=(1)), levels=np.linspace(22,30,21), extend='both', cmap='Spectral_r')
#xticks = np.arange(160, 280)
#axs.set_xticks(xticks)
axs.set_xlim(160, 280)
xticklabels = ['160°E', '180°W', '160°W', '140°W', '120°W', '100°W', '80°W']
axs.set_xticklabels(xticklabels)
colorbar = fig.colorbar(cf, ax=axs)
colorbar.set_label('SST [°C]')
axs.set_ylabel('Month')
axs.set_title('b) Climatological monthly mean SST (5S-5N) for PI$^{on}_{18}$')

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


# %%
