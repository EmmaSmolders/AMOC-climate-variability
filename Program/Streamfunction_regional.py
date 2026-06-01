#Program determines the MOV index

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
directory		= '/home/smolders/CESM_Collapse/Data/CESM/'

def ReadinData(filename):
	"""Read in the data"""
	fh = netcdf.Dataset(filename, 'r')

	lon 	= fh.variables['lon'][:]		#Longitude
	lat 	= fh.variables['lat'][10:86]		#Latitude 
	pres	= fh.variables['PS'][0, 10:86] * 0.01	#Surface pressure (hPa)
	hy_am	= fh.variables['hyam'][:]		#Hybrid A coefficient at layer mid-point
	hy_bm	= fh.variables['hybm'][:]		#Hybrid B coefficient at layer mid-point
	hy_ai	= fh.variables['hyai'][:]		#Hybrid A coefficient at layer interfaces
	hy_bi	= fh.variables['hybi'][:]		#Hybrid B coefficient at layer interfaces
	v_vel   = fh.variables['V'][0, :, 10:86] 	#Meridional velocity (m/s)
	#grid_x	= fh.variables['DX'][10:86]

	fh.close()
	
	fh 	= netcdf.Dataset(directory+'Atmosphere/Atmosphere_DX_DY_AREA.nc', 'r')
	grid_x	= fh.variables['DX'][10:86]
	fh.close()
	
	print(np.shape(lon))
	
	#Convert the grid to -180 to 180 grid
	lon_2, v_vel	= ConverterField3D(lon, v_vel)
	lon_2, grid_x	= ConverterField2D(lon, grid_x)
	lon, pres	= ConverterField2D(lon, pres)
	
	#lat_min_index	= (np.abs(lat - -5)).argmin()
	#lat_max_index	= (np.abs(lat - 5)).argmin()+1
	lon_min_index	= (np.abs(lon - -80)).argmin()
	lon_max_index	= (np.abs(lon - 15)).argmin()+1
	
	lon		= lon[lon_min_index:lon_max_index]
	grid_x		= grid_x[:, lon_min_index:lon_max_index]
	pres		= pres[:, lon_min_index:lon_max_index]
	v_vel		= v_vel[:, :, lon_min_index:lon_max_index]

	return lon, lat, grid_x, hy_am, hy_bm, hy_ai, hy_bi, pres, v_vel
	
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

files = glob.glob(directory_data+'*.cam2.h0.*.nc')
files.sort()

year_start	= 0
year_end	= 2200

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

#-----------------------------------------------------------------------------------------

#Empty array for all the zonal means
lon, lat, grid_x, hy_am, hy_bm, hy_ai, hy_bi, pres, v_vel = ReadinData(files[0])

print(lon)
print(np.shape(lon))
print(np.shape(grid_x))
print(lat)

#sys.exit()

#-----------------------------------------------------------------------------------------
#Get the pressure levels for pre-defined levels
P_lev		= np.asarray([4, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 925, 950, 975, 1000])
P_lev_log	= np.log(P_lev)
time_year	= ma.masked_all(int(len(time)/12))
stream_all	= ma.masked_all((len(time_year), len(P_lev), len(lat)))

for year_i in range(int(np.min(time)), int(np.min(time))+len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i - int(np.min(time))] = year_i
	files_month = glob.glob(directory_data+'b.e10.B1850.f19_g17.qe_hosing.001.cam2.h0.'+str(year_i).zfill(4)+'-*.nc')
	files_month.sort()

	stream_year		= ma.masked_all((12, len(P_lev), len(lat)))

	for file_i in range(len(files_month)):
		lon, lat, grid_x, hy_am, hy_bm, hy_ai, hy_bi, pres, v_vel	= ReadinData(files_month[file_i])

		#Empty arrays sigma levels
		P_3		= ma.masked_all((len(hy_am), len(lat), len(lon)))
		P_3_int		= ma.masked_all((len(hy_ai), len(lat), len(lon)))
		P_3_levels	= ma.masked_all((len(hy_am) * 2 - 1, len(lat), len(lon)))
		P_3_parts	= ma.masked_all((len(hy_am) * 2 - 1, len(lat), len(lon)))
		v_vel_parts	= ma.masked_all(shape(P_3_parts))
		stream_parts	= ma.masked_all(shape(P_3_parts))
		stream_function	= ma.masked_all((len(P_lev), len(lat), len(lon)))

		for lev_i in range(len(hy_am)):
			#Determine the pressure levels using the pressure at the surface
			P_3[lev_i]	= (hy_am[lev_i] * 10**3.0 + hy_bm[lev_i] * pres)

		for lev_i in range(len(hy_ai)):
			#Determine the pressure levels using the pressure at the surface
			P_3_int[lev_i]	= (hy_ai[lev_i] * 10**3.0 + hy_bi[lev_i] * pres)

		for lev_i in range(len(hy_am)):
			#Determine the differences between mid-point and upper boundary
			P_3_levels[lev_i * 2]	= P_3[lev_i]
			P_3_parts[lev_i * 2]	= (P_3[lev_i] - P_3_int[lev_i]) * 100.0
			v_vel_parts[lev_i * 2]	= v_vel[lev_i]

		for lev_i in range(len(hy_am) - 1):
			#Determine the differences between mid-point and lower boundary
			P_3_levels[lev_i * 2 + 1]	= P_3_int[lev_i + 1]
			P_3_parts[lev_i * 2 + 1]	= (P_3_int[lev_i + 1] - P_3[lev_i]) * 100.0
			v_vel_parts[lev_i * 2 + 1]	= v_vel[lev_i]

		for lev_i in range(len(P_3_parts)):
			#Determine the streamfunction
			if lev_i == 0:
				#First layer
				stream_parts[lev_i]	= P_3_parts[lev_i] * v_vel_parts[lev_i] * grid_x / 9.81
				
			else:
				#The remaining layers
				stream_parts[lev_i]	= (P_3_parts[lev_i] * v_vel_parts[lev_i] * grid_x / 9.81) + stream_parts[lev_i - 1]

		#Get to log (for interpolation)
		P_3_log	= np.log(P_3_levels)

		for lat_i in range(len(lat)):
			for lon_i in range(len(lon)):
				#Get the pressure coordinates for given point and interpolate
				x, y		= P_3_log[:, lat_i, lon_i], stream_parts[:, lat_i, lon_i]
				x_int		= P_lev_log[P_lev_log <= x[-1]]
				stream_int	= interp1d(x,y)(x_int)

				#Save field
				stream_function[:len(stream_int), lat_i, lon_i]	= stream_int
			
		#Take the zonal sum
		stream_year[file_i]	= np.sum(stream_function, axis = 2)

	#------------------------------------------------------------------------------
	month_days	= np.asarray([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])

	#Fill the array's with the same dimensions
	month_days_all	= ma.masked_all((len(month_days), len(P_lev), len(lat)))

	for month_i in range(len(month_days)):
		month_days_all[month_i]		= month_days[month_i]
		month_days_all[month_i]		= ma.masked_array(month_days_all[month_i], mask = stream_year[month_i].mask)

	month_days_all	= month_days_all / np.sum(month_days_all, axis = 0)

	#-----------------------------------------------------------------------------------------
	#Determine the time mean over the months of choice
	stream_all[year_i - int(np.min(time))]	= np.sum(stream_year * month_days_all, axis = 0)

#-----------------------------------------------------------------------------------------
print('Data is written to file')
fh = netcdf.Dataset(directory+'Atmosphere/Streamfunction_year_'+str(year_start)+'-'+str(year_end)+'_lon_80W_15E.nc', 'w')

fh.createDimension('time', len(time_year))
fh.createDimension('lev', len(P_lev))
fh.createDimension('lat', len(lat))

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('lev', float, ('lev'), zlib=True)
fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('SF', float, ('time', 'lev', 'lat'), zlib=True)

fh.variables['lev'].long_name		= 'Array of pressure levels'
fh.variables['lat'].long_name		= 'Array of latitudes'
fh.variables['SF'].long_name		= 'Atmospheric Streamfunction'

fh.variables['lev'].units 		= 'hPa'
fh.variables['lat'].units 		= 'Degrees N'
fh.variables['SF'].units 		= 'kg / s'

#Writing data to correct variable
fh.variables['time'][:] 		= time_year
fh.variables['lev'][:] 			= P_lev
fh.variables['lat'][:] 			= lat
fh.variables['SF'][:] 			= stream_all

fh.close()
