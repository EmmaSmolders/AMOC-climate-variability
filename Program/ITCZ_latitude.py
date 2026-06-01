#Program determines the latitude of the ITCZ in CESM QE simulation

from pylab import *
import numpy
import datetime
import time
import glob, os
import math
import netCDF4 as netcdf
from scipy import stats

#Making pathway to folder with all data
directory_data		= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing.001/OUTPUT/atm/hist/'
directory_data_600	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y600.001/OUTPUT/atm/hist/'
directory_data_1500	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y1500.001/OUTPUT/atm/hist/'
directory_data_2900	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y2900.001/run/'
directory_data_3800	= '/projects/0/prace_imau/prace_2013081679/cesm1_0_5/b.e10.B1850.f19_g17.qe_hosing_branched_off_y3800.001/OUTPUT/atm/hist/'
directory		= '/home/smolders/CESM_Collapse/Data/CESM/'

def ReadinData(filename, lat_min = -25, lat_max = 25):

	"""Read in the data"""
	fh = netcdf.Dataset(filename, 'r')

	lon 		= fh.variables['lon'][:]			#Longitude
	lat 		= fh.variables['lat'][:]			#Latitude 
	prec		= fh.variables['PRECT'][0]  * 1000.0 * 86400.0 	#Precipitation (mm day)
	long_wave	= fh.variables['FLUT'][0] * -1.0		#Outgoing longwave radiation at top of model (W/m^2)

	fh.close()

	lat_min_index	= (fabs(lat - lat_min)).argmin()
	lat_max_index	= (fabs(lat - lat_max)).argmin() + 1
	lat		= lat[lat_min_index:lat_max_index]
	prec		= prec[lat_min_index:lat_max_index]
	long_wave	= long_wave[lat_min_index:lat_max_index]

	return lon, lat, prec, long_wave

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

def ConverterField2D(lon, field):
	"""Shifts field, to -180E to 180E"""
	lon_new		= ma.masked_all(shape(lon))
	field_new	= ma.masked_all(shape(field))

	#Get the corresponding index
	index		= (fabs(lon - 180)).argmin()

	#Start filling at -180
	lon_new[:len(lon[index:])] 		= lon[index:]
	field_new[:, :len(lon[index:])]		= field[:, index:]

	#Fill the remaining part
	lon_new[len(lon[index:]):] 		= lon[:index]
	field_new[:, len(lon[index:]):]		= field[:, :index]

	lon_new[lon_new >= 179.9]		= lon_new[lon_new >= 179.9] - 360.0

	return lon_new, field_new

def PeriodicBoundaries3D(lon, lat, field, lon_grids = 1):
        """Add periodic zonal boundaries for 2D field"""

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

def PeriodicBoundaries2D(lon, lat, field, lon_grids = 1):
        """Add periodic zonal boundaries for 2D field"""

        #Empty field with additional zonal boundaries
        lon_2                   = np.zeros(len(lon) + lon_grids * 2)
        field_2                 = ma.masked_all((len(lat), len(lon_2)))

        #Get the left boundary, which is the right boundary of the original field
        lon_2[:lon_grids]       = lon[-lon_grids:] - 360.0
        field_2[:, :lon_grids]  = field[:, -lon_grids:]

        #Same for the right boundary
        lon_2[-lon_grids:]      = lon[:lon_grids] + 360.0
        field_2[:, -lon_grids:] = field[:, :lon_grids]

        #And the complete field
        lon_2[lon_grids:-lon_grids]             = lon
        field_2[:, lon_grids:-lon_grids]        = field

        return lon_2, field_2

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------

month_start	= 1
month_end	= 12

year_start	= 4199
year_end	= 4300

#-----------------------------------------------------------------------------------------

files = glob.glob(directory_data_3800+'*.cam2.h0.*.nc')
files.sort()

print(files[0])
print(files[-1])

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

print(time)

#sys.exit()

#-----------------------------------------------------------------------------------------

#Determine the section length per depth layer
lon, lat, prec, long_wave	= ReadinData(files[0])
time_year			= ma.masked_all(int(len(time)/12))
time_month			= ma.masked_all(len(time))
ITCZ_all			= ma.masked_all((len(time_year), len(lat), len(lon)))
ITCZ_month			= ma.masked_all((len(time), len(lat), len(lon)))

for year_i in range(int(np.min(time)), int(np.min(time))+len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i - int(np.min(time))] = year_i
	#files_month = glob.glob(directory_data_3800+'b.e10.B1850.f19_g17.qe_hosing.001.cam2.h0.'+str(year_i).zfill(4)+'-*.nc')
	#files_month.sort()

	ITCZ_year		= ma.masked_all((12, len(lat), len(lon)))

	for month_i in range(month_start, month_end+1):
		#Loop over each month
		print(month_i)
	
		filename = files[0][:-10]+str(year_i).zfill(4)+'-'+str(month_i).zfill(2)+'.nc'
		print(filename)
	
		lon, lat, prec, long_wave	= ReadinData(filename)

		#Now get the zonal means over the lon width (= 15.0), first take half
		lon_width_index = (fabs(lon - lon[0] - 15.0 / 2.0)).argmin()

		#Get periodic boundaries
		lon_2, prec	= PeriodicBoundaries2D(lon, lat, prec, lon_width_index)
		lon_2, long_wave= PeriodicBoundaries2D(lon, lat, long_wave, lon_width_index)

		prec_lon	= ma.masked_all((len(lat), len(lon)))
		long_wave_lon	= ma.masked_all((len(lat), len(lon)))

		for lon_i in range(lon_width_index, len(lon_2) - lon_width_index):
			#Take the zonal mean over the lon bins
			prec_lon[:, lon_i - lon_width_index]	 = np.mean(prec[:, lon_i - lon_width_index:lon_i + lon_width_index+1], axis = 1)
			long_wave_lon[:, lon_i - lon_width_index]= np.mean(long_wave[:, lon_i - lon_width_index:lon_i + lon_width_index+1], axis = 1)

		#Now determine the ITCZ position
		num_lat	= float(len(lat))

		for lon_i in range(len(lon)):
			#Get each Joint Cummulative Distribution Function (J-CDF)
			for lat_i in range(len(lat)):
				#Determine J-CDF
				index_min	= np.where((prec[:, lon_i] <= prec[lat_i, lon_i]) & (long_wave[:, lon_i] <= long_wave[lat_i, lon_i]))[0]

				#Determine the probability 
				prob		= len(index_min) / num_lat

				if prob >= 0.85:
					#Location of the ITCZ
					ITCZ_year[month_i - month_start, lat_i, lon_i] = 1.0
					
		ITCZ_month[(year_i - int(np.min(time)))*12 + month_i - month_start, :, :] = ITCZ_year[month_i - month_start, :, :]

	#-----------------------------------------------------------------------------------------
	month_days	= np.asarray([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31., 31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
	month_days	= month_days[month_start-1:month_end]

	#Fill the array's with the same dimensions
	month_days_all	= ma.masked_all((len(month_days), len(lat), len(lon)))

	for month_i in range(len(month_days)):
		month_days_all[month_i]		= month_days[month_i]
		month_days_all[month_i]		= ma.masked_array(month_days_all[month_i], mask = ITCZ_year[month_i].mask)

	month_days_all	= month_days_all / np.sum(month_days_all, axis = 0)

	#-----------------------------------------------------------------------------------------
	
	#Determine the total frequency per year
	ITCZ_all[year_i - int(np.min(time))]		= np.sum(ITCZ_year * month_days_all, axis = 0) / 12.0

#Convert the grid to -180 to 180
lon_2, ITCZ_all		= ConverterField3D(lon, ITCZ_all)
lon_2, ITCZ_month	= ConverterField3D(lon, ITCZ_month)
lon, ITCZ_month		= PeriodicBoundaries3D(lon_2, lat, ITCZ_month)
lon, ITCZ_month		= lon[1:], ITCZ_month[:, :, 1:]
lon, ITCZ_all		= PeriodicBoundaries3D(lon_2, lat, ITCZ_all)
lon, ITCZ_all		= lon[1:], ITCZ_all[:, :, 1:]

#-----------------------------------------------------------------------------------------
print('Data is written to file')
fh = netcdf.Dataset(directory+'Atmosphere/ITCZ_branch3800_year_'+str(year_start)+'-'+str(year_end)+'.nc', 'w')

fh.createDimension('time', len(time_year))
fh.createDimension('time_month', len(time))
fh.createDimension('lat', len(lat))
fh.createDimension('lon', len(lon))

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('time_month', float, ('time_month'), zlib=True)
fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('lon', float, ('lon'), zlib=True)
fh.createVariable('ITCZ', float, ('time', 'lat', 'lon'), zlib=True)
fh.createVariable('ITCZ_month', float, ('time_month', 'lat', 'lon'), zlib=True)

fh.variables['lon'].longname 		= 'Array of longitudes'
fh.variables['lat'].longname 		= 'Array of latitudes'
fh.variables['ITCZ'].longname 		= 'Yearly frequency of ITCZ'
fh.variables['ITCZ_month'].longname 	= 'Monthly frequency of ITCZ'

fh.variables['time'].units 		= 'year'
fh.variables['time_month'].units 	= 'month'
fh.variables['lon'].units 		= 'Degrees E'
fh.variables['lat'].units 		= 'Degrees N'

#Writing data to correct variable
fh.variables['time'][:] 		= time_year	
fh.variables['time_month'][:] 		= time
fh.variables['lon'][:] 			= lon
fh.variables['lat'][:] 			= lat
fh.variables['ITCZ'][:] 		= ITCZ_all
fh.variables['ITCZ_month'][:] 		= ITCZ_month

fh.close()
