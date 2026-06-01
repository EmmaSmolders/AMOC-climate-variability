#Program determines the meridional heat transport

from pylab import *
import numpy
import datetime
import time
import glob, os
import math
import netCDF4 as netcdf
import matplotlib.colors as colors
from scipy import stats

#Making pathway to folder with all data
directory_data		= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y600.001/OUTPUT/ocn/hist/monthly/'
directory		= '/home/rvwesten/MOV/Data/LR-CESM_0600/'

def ReadinData(filename):

	fh = netcdf.Dataset(filename, 'r')
	
	lat			= fh.variables['lat_aux_grid'][:]
	MHT_global		= fh.variables['N_HEAT'][0, 0, 0]
	
	return lat, MHT_global

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------

year_start	= 1000
year_end	= 1100

#-----------------------------------------------------------------------------------------
files = glob.glob(directory_data+'*pop.h.*.nc')
files.sort()

#-----------------------------------------------------------------------------------------

#Define empty array's
time_all 	= np.zeros(len(files))

for year_i in range(len(files)):
	date  = files[year_i][-10:-3]	
	year  = int(date[0:4])
	month = int(date[5:7])

	time_all[year_i] = year + (month-1) / 12.0

time_start	= (np.abs(time_all - year_start)).argmin()
time_end	= (np.abs(time_all - (year_end))).argmin()+12

time_all	= time_all[time_start:time_end]
files		= files[time_start:time_end]

print(files[0])
print(files[-1])

#-----------------------------------------------------------------------------------------
lat, MHT	= ReadinData(files[0])
time_year	= np.zeros(int(len(time_all) / 12))
MHT_all		= ma.masked_all((len(time_year), len(lat)))

for year_i in range(len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i] 	= int(time_all[year_i*12])
	files_month 		= files[year_i*12:(year_i+1)*12]

	MHT			= ma.masked_all((12, len(lat)))
	
	for month_i in range(len(files_month)):
		print(files_month[month_i])
		lat, MHT[month_i]	= ReadinData(files_month[month_i])
		
	#------------------------------------------------------------------------------
	month_days	= np.asarray([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
	month_days	= month_days / np.sum(month_days)

	#Fill the array's with the same dimensions
	month_days_all	= ma.masked_all((len(month_days), len(lat)))

	for month_i in range(len(month_days)):
		month_days_all[month_i]		= month_days[month_i]

	#-----------------------------------------------------------------------------------------

	#Determine the time mean over the months of choice
	MHT_all[year_i]	= np.sum(MHT * month_days_all, axis = 0)

#-----------------------------------------------------------------------------------------

print('Data is written to file')
fh = netcdf.Dataset(directory+'/Ocean/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'.nc', 'w')

fh.createDimension('lat', len(lat))

fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('MHT', float, ('lat'), zlib=True)

fh.variables['lat'].longname 			= 'Array of latitudes'
fh.variables['MHT'].longname 			= 'Meridional heat transport'
	
fh.variables['lat'].units 			= 'degrees N'
fh.variables['MHT'].units 			= 'PW'

#Writing data to correct variable	
fh.variables['lat'][:] 				= lat
fh.variables['MHT'][:] 				= np.mean(MHT_all, axis = 0)

fh.close()
