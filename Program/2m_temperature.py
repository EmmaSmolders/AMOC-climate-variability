#Program retrieves sea level pressure 

from pylab import *
import numpy
import datetime
import time
import glob, os
import math
import netCDF4 as netcdf
from scipy import stats
from scipy.interpolate import interp1d

#Making pathway to folder with all data
directory_data		= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing.001/OUTPUT/atm/hist/'
directory_data_600	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y600.001/OUTPUT/atm/hist/'
directory_data_1500	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y1500.001/OUTPUT/atm/hist/'
directory_data_2900	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y2900.001/run/'
directory_data_3800	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y3800.001/OUTPUT/atm/hist/'
directory		= '/home/smolders/CESM_Collapse/Data/CESM/'

def ReadinData(filename):
	"""Read in the data"""
	fh = netcdf.Dataset(filename, 'r')

	lon 	= fh.variables['lon'][:]		#Longitude
	lat 	= fh.variables['lat'][:]		#Latitude 
	TS	= fh.variables['TREFHT'][0, :]      	#Reference height temperature [K] (lat,lon)

	fh.close()
	
	print(np.shape(lon))
	print(np.shape(lat))
	print(np.shape(TS))
	
	#sys.exit()

	#Convert the grid to -180 to 180 grid
	lon, TS	= ConverterField2D(lon, TS)
	
	#lat_min_index	= (np.abs(lat - -5)).argmin()
	#lat_max_index	= (np.abs(lat - 5)).argmin()+1
	#lon_min_index	= (np.abs(lon - -80)).argmin()
	#lon_max_index	= (np.abs(lon - 15)).argmin()+1
	
	#lon		= lon[lon_min_index:lon_max_index]
	#lat		= lat[lat_min_index:lat_max_index]
	#TS		= TS[lat_min_index:lat_max_index, lon_min_index:lon_max_index]

	return lon, lat, TS

def ConverterField2D(lon, field):
	"""Shifts field, to -180E to 180E"""
	lon_new		= ma.masked_all(shape(lon))
	field_new	= ma.masked_all(shape(field))

	#Get the corresponding index
	index		= (fabs(lon - 180)).argmin()

	#Start filling at -180
	lon_new[:len(lon[index:])] 	= lon[index:]
	field_new[:, :len(lon[index:])]	= field[:, index:]

	#Fill the remaining part
	lon_new[len(lon[index:]):] 	= lon[:index]
	field_new[:, len(lon[index:]):]	= field[:, :index]

	lon_new[lon_new >= 179.9]	= lon_new[lon_new >= 179.9] - 360.0

	return lon_new, field_new
	
def ConverterField3D(lon, field):
	"""Shifts field, to -180E to 180E"""
	lon_new		= ma.masked_all(shape(lon))
	field_new	= ma.masked_all(shape(field))

	#Get the corresponding index
	index		= (fabs(lon - 180)).argmin()

	#Start filling at -180
	lon_new[:len(lon[index:])] 		= lon[index:]
	field_new[:, :, :len(lon[index:])]	= field[:, :, index:]

	#Fill the remaining part
	lon_new[len(lon[index:]):] 		= lon[:index]
	field_new[:, :, len(lon[index:]):]	= field[:, :, :index]

	lon_new[lon_new >= 179.9]		= lon_new[lon_new >= 179.9] - 360.0

	return lon_new, field_new

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------

month_start	= 1
month_end	= 12

year_start	= 999
year_end	= 1100

#-----------------------------------------------------------------------------------------

files = glob.glob(directory_data_600+'*.cam2.h0.*.nc')
files.sort()

#-----------------------------------------------------------------------------------------

#Define empty array's
time 		= np.zeros(len(files))

for year_i in range(len(files)):	
	date  = files[year_i][-10:-3]	
	year  = int(date[0:4])
	month = int(date[5:7])

	time[year_i] = year + (month-1) / 12.0

time_start	= (np.abs(time - year_start)).argmin()
time_end	= (np.abs(time - (year_end+1))).argmin()

time		= time[time_start:time_end]
files		= files[time_start:time_end]

print(files[0])
print(files[-1])

sys.exit()

#-----------------------------------------------------------------------------------------

#Empty array for all the zonal means
lon, lat, TS = ReadinData(files[0])

print(np.shape(lon))
print(np.shape(lat))
print(lon)
print(lat)

#sys.exit()

time_year		= ma.masked_all(int(len(time)/12))
TS_all			= ma.masked_all((len(time_year), len(lat), len(lon)))

for year_i in range(int(np.min(time)), int(np.min(time))+len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i - int(np.min(time))] = year_i

	TS_year		= ma.masked_all((month_end-month_start+1, len(lat), len(lon)))
	
	for month_i in range(month_start, month_end+1):
		#Loop over each month

		if month_i <= 12:
			#Same year
			year_j	= np.copy(year_i)
			month_j	= np.copy(month_i)
		else:
			#Next year
			year_j	= year_i+1
			month_j	= month_i - 12

		filename = files[0][:-10]+str(year_j).zfill(4)+'-'+str(month_j).zfill(2)+'.nc'
		
		print(filename)

		lon, lat, TS	= ReadinData(filename)
		
		TS_year[month_i - month_start, :, :] = TS
				
	#------------------------------------------------------------------------------
	month_days	= np.asarray([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31., 31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
	month_days	= month_days[month_start-1:month_end]
	
	print(month_days)

	#Fill the array's with the same dimensions
	month_days_all	= ma.masked_all((len(month_days), len(lat), len(lon)))

	for month_i in range(len(month_days)):
		month_days_all[month_i]		= month_days[month_i]
		month_days_all[month_i]		= ma.masked_array(month_days_all[month_i], mask = TS_year[month_i].mask)

	month_days_all	= month_days_all / np.sum(month_days_all, axis = 0)

	#-----------------------------------------------------------------------------------------

	#Determine the time mean over the months of choice
	TS_year	= np.sum(TS_year * month_days_all, axis = 0)

	#
	TS_all[year_i - int(np.min(time))]	= TS_year

#-----------------------------------------------------------------------------------------

print('Data is written to file')
fh = netcdf.Dataset(directory+'Atmosphere/2m_TEMP_month_'+str(month_start)+'-'+str(month_end)+'_brach2900_year_'+str(year_start)+'-'+str(year_end)+'.nc', 'w')

fh.createDimension('time', len(time_year))
fh.createDimension('lat', len(lat))
fh.createDimension('lon', len(lon))

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('lon', float, ('lon'), zlib=True)
fh.createVariable('TS', float, ('time', 'lat', 'lon'), zlib=True)

fh.variables['lat'].long_name		= 'Array of latitudes'
fh.variables['lon'].long_name		= 'Array of longitudes'
fh.variables['TS'].long_name		= 'Time mean surface reference air temperature'

fh.variables['time'].units 		= 'Model year'
fh.variables['lat'].units 		= 'Degrees N'
fh.variables['lon'].units 		= 'Degrees E'
fh.variables['TS'].units 		= 'K'

#Writing data to correct variable
fh.variables['time'][:] 		= time_year
fh.variables['lat'][:] 			= lat
fh.variables['lon'][:] 			= lon
fh.variables['TS'][:] 			= TS_all

fh.close()


