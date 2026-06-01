#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:58:53 2025

@author: 6008399

EOF PDV

"""

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
directory_data	= '/home/smolders/CESM_Collapse/Data/CESM/Ocean/'

def Inside_polygon(x_point, y_point, corners):
    """Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies corners = [(x1, y1), (x2, x2), ... , (xN, yN)]. Note that you do NOT have
    to insert the first vertices as the last element"""

    n = len(corners) #Number of points of polygon
    inside = False  #Assume that point is not in polygon
    p1x, p1y = corners[0] #Take the first points of polygon
    p1x, p1y = float(p1x), float(p1y) #Make floats, otherwise program does not work (if there are any integers as input)
    
    for ver_i in range(1, n + 1):     #Extra number, to make sure that the first one is also taken into account (closed loop)
        p2x, p2y = corners[ver_i % n] #Read out the next point, and in the end also the first point again
        if y_point > min(p1y, p2y):
            if y_point <= max(p1y, p2y):
                if x_point <= max(p1x, p2x):
                    if p1y != p2y:
                    	#The point should be left above a vertex, but in between the y values of the vertices
                    	#This can be checked while determining the intersection of two lines
                    	#1) A horizontal line through the point of interest, so y = y_point
                        #2) The line of a single vertex
                        x_intersection = (y_point - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x_point <= x_intersection:
                    	#This implies, if the point is indeed left (above) of the vertex, say that point is in polygon
                    	#However, it might be possible that the polygon has a particular shape,
                    	#So a second encounter in this line says that it is false again.
                        inside = not inside
        p1x, p1y = p2x, p2y #Save the old one as the first one, and continue
    return inside

def Points(lon_domain, lat_domain):
	"""Returns the coordinates of the domain, which are used to determine the mask of the different ocean basins"""

	points = []

	for point_i in range(len(lon_domain) - 1): #Do not include any double elements
		points.append((float(lon_domain[point_i]), float(lat_domain[point_i])))

	return points

#%%

#Choose which months you want (i.e. DJF or JJA)
month_start = 1
month_end   = 12

#fh      = netcdf.Dataset(directory_data+'SST_month_'+str(month_start)+'-'+str(month_end)+'_QE_year_0-2200.nc', 'r')

fh       = netcdf.Dataset(directory_data + 'SST_Pacific_year_600-1500_month_1-12_QE.nc', 'r')

time_month_forward      = fh.variables['time_month'][:]     #Model years
lon                     = fh.variables['lon'][200::,:]            #Array of longitudes [degE]
lat                     = fh.variables['lat'][200::,:]            #Array of latitudes [degN]
area                    = fh.variables['area'][200::,:]
SST_month_forward       = fh.variables['SST_month'][:,200::,:]      #Sea level pressure (av\eraged over months) [hPa]

fh.close()

#fh       = netcdf.Dataset(directory_data + 'SST_Pacific_year_2900-3800_month_1-12_QE.nc', 'r')
fh       = netcdf.Dataset(directory_data + 'SST_Pacific_year_2900-3800_month_1-12_QE.nc', 'r')

time_month_backward      = fh.variables['time_month'][:]     #Model years
lon                     = fh.variables['lon'][200::,:]            #Array of longitudes [degE]
lat                     = fh.variables['lat'][200::,:]            #Array of latitudes [degN]
area                    = fh.variables['area'][200::,:]
SST_month_backward       = fh.variables['SST_month'][:,200::,:]      #Sea level pressure (av\eraged over months) [hPa]

fh.close()



#%% Functions

#Central moving average
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def TrendRemover(time, data, trend_type):
	"""Removes trend of choice"""
	
	rank = polyfit(time, data, trend_type)
	fitting = 0.0 
		
	for rank_i in range(len(rank)):
			
		fitting += rank[rank_i] * (time**(len(rank) - 1 - rank_i))

	data -= fitting
	
	return data

def MonthRemover(time, data_all):
    """Removes the monthly average for each time series"""
    
    for month_i in range(12):
        month_index             = np.arange(month_i, len(time), 12)
        data_all[month_index]   = data_all[month_index] - np.mean(data_all[month_index], axis = 0)

    return data_all

def MovingAverage(time, data, moving_average):
	"""Determines moving average of time series"""
	
	#Empty array's for the smoothed time series
	time_2     = ma.masked_all(len(time) - moving_average + 1)
	data_2 	   = ma.masked_all(len(time_2))
	
	#Determine the so-called middle index where the moving average is determined
	selection  = (moving_average - 1) / 2		

	for time_i in range(selection, selection + len(time_2)):
		#Take moving average
		data_2[time_i - selection]  = np.mean(data[time_i-selection:time_i-selection + moving_average], axis = 0)
		time_2[time_i - selection] 	= time[time_i]
	
	return time_2, data_2

def Distance(lon1, lat1, lon2, lat2):
	"""Returns distance (m) of two points located at the globe
	coordinates need input in degrees"""

	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) #Convert to radians

	#Haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
	c = 2.0 * math.asin(sqrt(a)) 
	r = 6371000.0 # Radius of earth in meters
	
	return c * r #Distance between two points in meter

def AreaComputer(longitude, latitude, field):
    """Determines the area (m^2) per grid cell
    returns 2-D array (lat, lon) with the area per box"""

    #Define empty array for latitude per grid cell and the Area covered by the Ocean
    area_grids  = ma.masked_all(np.shape(field))

    for lat_i in range(len(latitude)):
        for lon_i in range(len(longitude[0])):

            #Determine latitude of grid cell
            if lat_i == 0:  #Lower boundary, extrapolate for other boundary
                lat2 = (latitude[lat_i, lon_i] + latitude[lat_i + 1, lon_i])/2.0
                lat1 = latitude[lat_i, lon_i] - (lat2 - latitude[lat_i, lon_i])

            elif lat_i != len(latitude) - 1: #Take the boundaries of the grid cell
                lat1 = (latitude[lat_i - 1, lon_i] + latitude[lat_i, lon_i])/2.0
                lat2 = (latitude[lat_i, lon_i] + latitude[lat_i + 1, lon_i])/2.0

            else:   #Upper boundary, extrapolate for other boundary
                lat1 = (latitude[lat_i - 1, lon_i] + latitude[lat_i, lon_i])/2.0
                lat2 = latitude[lat_i, lon_i] + (latitude[lat_i, lon_i] - lat1)

            #Determining zonal length (m), is latitude dependent, therefore, take middle of grid cell
            length_zonal_grid       = Distance(0.0, latitude[lat_i, lon_i], np.mean(np.diff(longitude[lat_i, :])), latitude[lat_i, lon_i]) 
            #Determining meriodinal length (m), is longitude independent
            length_meridional_grid  = Distance(0.0, lat1, 0.0, lat2)    

            area_grids[lat_i, lon_i] = length_zonal_grid * length_meridional_grid

    try:
        #Set everything to land mask
        area_grids  = ma.masked_array(area_grids, mask = field.mask)

    except:
        pass  

    return area_grids

#%%


moving_average = 0

# Define a function for EOF analysis
def perform_eof_analysis(SST, time, lat, lon, area, trend_type=2, remove_month=1, moving_average=0, eigen_number=1, total_variance=90):
    """
    Perform EOF analysis on the given SST dataset.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()

    # Plot the time mean of the data
    c = ax.contourf(lon,lat, SST.mean(axis=0), transform=ccrs.PlateCarree(), cmap='YlGnBu')

    # Add a colorbar
    plt.colorbar(c, ax=ax, orientation='horizontal')

    # Add some titles
    #plt.title('Mean surface level pressure (model year '+str(time_month_1)+' - '+str(time2)+')')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.show()
    
    # Area-weighted average
    area_grids = area
    area_grids = ma.masked_array(area_grids, mask=SST[0].mask)
    area_grids = area_grids / area_grids.max()  # Normalize area

    # Pre-processing
    if trend_type > 0:
        print('Trend is removed\n')
        for lat_i in range(len(lat)):
            for lon_i in range(len(lon[0])):
                SST[:, lat_i, lon_i] = TrendRemover(time, SST[:, lat_i, lon_i], trend_type)
                
    plt.figure()
    plt.contourf(lon,lat, SST.mean(axis=0))
    plt.show()

    if remove_month == 1:
        print('Monthly signal is removed\n')
        SST = MonthRemover(time, SST)
        
    plt.figure()
    plt.contourf(lon,lat, SST.mean(axis=0))
    plt.show()

    if moving_average > 1:
        print('Moving average is applied\n')
        time_2 = np.zeros((len(time) - moving_average + 1))
        SST_2 = ma.masked_all((len(time_2), len(lat), len(lon[0])))

        selection = int((moving_average - 1) / 2)

        for time_i in range(int(selection), int(selection + len(time_2))):
            time_2[time_i - selection] = time[time_i]
            SST_2[time_i - selection] = np.mean(SST[time_i - selection:time_i - selection + moving_average], axis=0)

        time = time_2
        SST = SST_2
        del time_2, SST_2

    print('Data is normalized\n')
    SST = SST - np.mean(SST, axis=0)
    SST = SST / np.std(SST, axis=0)
    
    plt.figure()
    plt.contourf(lon,lat, SST.mean(axis=0))
    plt.show()

    print('Data is scaled by area\n')
    SST = SST * area_grids
    
    print(np.shape(SST))
    print(np.shape(area_grids))
    
    plt.figure()
    plt.contourf(lon,lat, area_grids)
    plt.show()
    
    plt.figure()
    plt.contourf(lon,lat, SST.mean(axis=0))
    plt.show()

    # EOF analysis
    masked_field = SST[0].mask
    data_all = ma.masked_all((len(lon[0]) * len(lat) - sum(masked_field), len(time)))
    grid_counter = 0

    for lat_i in range(len(lat)):
        for lon_i in range(len(lon[0])):
            if masked_field[lat_i, lon_i] == False:
                data_all[grid_counter] = SST[:, lat_i, lon_i]
                grid_counter += 1

    print('Determining EOFs and PCs\n')
    u1, s1, v1 = np.linalg.svd(data_all)
    
    print(s1)
    print(u1)
    print(v1)

    # Determine the variance
    s1 = s1 ** 2.0
    print(eigen_number)
    print(s1)
    print('Eigenvector number', eigen_number, 'contributes ', round(s1[eigen_number - 1] / sum(s1) * 100.0, 1), '% of the total variance\n')

    variance = 0.0
    for eigen_i in range(len(u1)):
        variance += s1[eigen_i] / sum(s1) * 100.0
        if round(variance, 1) >= total_variance:
            print(total_variance)
            print(variance)
            print(len(u1))
            print('Number of EOFs needed to include at least ' + str(total_variance) + '% of the variance:', eigen_i, '\n')
            break

    # Retrieve the EOFs
    eof = ma.masked_all((5, len(lat), len(lon[0])))
    grid_counter = 0

    for lat_i in range(len(lat)):
        for lon_i in range(len(lon[0])):
            if masked_field[lat_i, lon_i] == False:
                eof[:, lat_i, lon_i] = u1[grid_counter, :len(eof)]
                grid_counter += 1

    return eof, u1, s1, v1, time

#%%
# Process all SST datasets
#lon1, lon2 = -90, 30
#lat1, lat2 = 0, 80
#time1, time2 = 0, 2200
#depth_level = 500

#lat_min_index  = (np.abs(lat[:,0] - lat1)).argmin()
#lat_max_index  = (np.abs(lat[:,0] - lat2)).argmin()+1
#lon_min_index   = (np.abs(lon[0,:] - lon1)).argmin()
#lon_max_index   = (np.abs(lon[0,:] - lon2)).argmin()+1

datasets = {
    "SST_forward": (SST_month_forward, time_month_forward), 
    "SST_backward": (SST_month_backward, time_month_backward)}

results = {}

for name, (SST, time) in datasets.items():
    print(f"Processing {name}...")
    print(np.shape(SST))
    print(np.shape(time))
    print(np.shape(lat))
    print(np.shape(lon))
    print(np.shape(area))
    eof, u1, s1, v1, time = perform_eof_analysis(SST, time, lat, lon, area)
    results[name] = {
        "eof": eof,
        "u1": u1,
        "s1": s1,
        "v1": v1,
    }
    print(f"Finished processing {name}.\n")

    # Save the results to a NetCDF file
    filename = f"{directory_data}EOF_PDV_{name}_month_{month_start}_{month_end}_moving_average_{moving_average}_CESM_QE_year_{int(time[0])}_{int(time[-1])}_quadratic_detrend.nc"
    print(f"Saving results to {filename}...")

    fh = netcdf.Dataset(filename, 'w')
    fh.createDimension('lon', len(lon[0]))
    fh.createDimension('lat', len(lat))
    fh.createDimension('eof', len(eof))
    fh.createDimension('time', len(time))

    fh.createVariable('lon', float, ('lat', 'lon'), zlib=True)
    fh.createVariable('lat', float, ('lat', 'lon'), zlib=True)
    fh.createVariable('eof', float, ('eof'), zlib=True)
    fh.createVariable('time', float, ('time'), zlib=True)
    fh.createVariable('PC', float, ('eof', 'time'), zlib=True)
    fh.createVariable('VAR', float, ('eof'), zlib=True)
    fh.createVariable('EOF', float, ('eof', 'lat', 'lon'), zlib=True)

    fh.variables['lon'].longname = 'Array of longitudes'
    fh.variables['lat'].longname = 'Array of latitudes'
    fh.variables['PC'].long_name = 'PCs of the salinity'
    fh.variables['VAR'].long_name = 'Variance of the PCs/EOFS'
    fh.variables['EOF'].long_name = 'EOFs of the salinity'

    fh.variables['time'].units = 'Model year'
    fh.variables['lon'].units = 'Degrees east'
    fh.variables['lat'].units = 'Degrees north'
    fh.variables['VAR'].units = '%'

    # Writing data to correct variable
    fh.variables['lon'][:] = lon
    fh.variables['lat'][:] = lat
    fh.variables['time'][:] = time
    fh.variables['eof'][:] = np.arange(len(eof)) + 1
    fh.variables['PC'][:] = v1[:len(eof)]
    fh.variables['VAR'][:] = s1[:len(eof)] / sum(s1) * 100.0
    fh.variables['EOF'][:] = eof

    fh.close()
    print(f"Results saved to {filename}.\n")


