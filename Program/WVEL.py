#Program determines the meridional mean (5S-5N) vertical and horizontal velocities 

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
	pres	= fh.variables['PS'][0, :] * 0.01	#Surface pressure (hPa)
	hy_am	= fh.variables['hyam'][:]		#Hybrid A coefficient at layer mid-point
	hy_bm	= fh.variables['hybm'][:]		#Hybrid B coefficient at layer mid-point
	hy_ai	= fh.variables['hyai'][:]		#Hybrid A coefficient at layer interface
	hy_bi	= fh.variables['hybi'][:]		#Hybrid B coefficient at layer interface
	w_vel   = fh.variables['OMEGA'][0, :, :] 	#Vertical velocity (Pa/s)
	ww_vel  = fh.variables['OMEGA2'][0,:,:]
	u_vel 	= fh.variables['U'][0, :, :] 		#Zonal velocity (m/s)
	uu_vel  = fh.variables['UU'][0,:,:]
	v_vel 	= fh.variables['V'][0, :, :] 		#Meridional velocity (m/s)
	lat_weight 	= fh.variables['gw'][:]

	fh.close()
	
	print(np.shape(uu_vel))
	print(np.shape(hy_am))
	print(np.shape(lon))
	print(np.shape(lat))
	print(np.shape(lat_weight))
	
	#sys.exit()
	
	#print(np.shape(lat_weight))
	#print(np.shape(lat))

	#Convert the grid to -180 to 180 grid
	#lon_2, u_vel	= ConverterField3D(lon, u_vel)
	#lon_2, v_vel	= ConverterField3D(lon, v_vel)
	#lon_2, w_vel	= ConverterField3D(lon, w_vel)
	#lon, pres	= ConverterField2D(lon, pres)
	
	#print(lon)
	
	#Take equatorial region (-5 - 5N)
	lat_min_index	= (np.abs(lat - 1)).argmin()
	lat_max_index	= (np.abs(lat - 5)).argmin()+1
	#lon_min_index	= (np.abs(lon - -80)).argmin()
	#lon_max_index	= (np.abs(lon - 15)).argmin()+1
	
	#lon		= lon[lon_min_index:lon_max_index]
	lat		= lat[lat_min_index:lat_max_index]
	lat_weight 	= lat_weight[lat_min_index:lat_max_index]
	pres		= pres[lat_min_index:lat_max_index, :]
	u_vel		= u_vel[:, lat_min_index:lat_max_index, :]
	v_vel		= v_vel[:, lat_min_index:lat_max_index, :]
	w_vel		= w_vel[:, lat_min_index:lat_max_index, :]

	print(lat)
	
	sys.exit()
	return lon, lat, lat_weight, hy_am, hy_bm, hy_ai, hy_bi, pres, u_vel, v_vel, w_vel

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

year_start	= 4199
year_end	= 4300

#-----------------------------------------------------------------------------------------

files = glob.glob(directory_data_3800+'*.cam2.h0.*.nc')
files.sort()

print(len(files))

print(files[0])
print(files[-1])

#sys.exit()

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

#Empty array for all the zonal means
lon, lat, lat_weight, hy_am, hy_bm, hy_ai, hy_bi, pres, u_vel, v_vel, w_vel = ReadinData(files[0])

#Get the pressure levels for pre-defined levels
P_lev		= np.asarray([4, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 925, 950, 975, 1000])
P_lev_log	= np.log(P_lev)
time_year	= ma.masked_all(int(len(time)/12))

u_vel_all	= ma.masked_all((len(time_year), len(P_lev), len(lon)))
v_vel_all	= ma.masked_all((len(time_year), len(P_lev), len(lon)))
w_vel_all	= ma.masked_all((len(time_year), len(P_lev), len(lon)))

for year_i in range(int(np.min(time)), int(np.min(time))+len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i - int(np.min(time))] = year_i

	u_vel_year		= ma.masked_all((month_end-month_start+1, len(P_lev), len(lat), len(lon)))
	v_vel_year		= ma.masked_all((month_end-month_start+1, len(P_lev), len(lat), len(lon)))
	w_vel_year		= ma.masked_all((month_end-month_start+1, len(P_lev), len(lat), len(lon)))
	
	for month_i in range(month_start, month_end+1):
		#Loop over each month
		#print(month_i)

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

		lon, lat, lat_weight, hy_am, hy_bm, hy_ai, hy_bi, pres, u_vel, v_vel, w_vel	= ReadinData(filename)

		#Empty arrays sigma levels
		P_3		= ma.masked_all((len(hy_am), len(lat), len(lon)))
		P_3_int		= ma.masked_all((len(hy_ai), len(lat), len(lon)))

		for lev_i in range(len(hy_am)):
			#Determine the pressure levels using the pressure at the surface
			P_3[lev_i]	= (hy_am[lev_i] * 10**3.0 + hy_bm[lev_i] * pres)

		for lev_i in range(len(hy_ai)):
			#Determine the pressure levels using the pressure at the surface
			P_3_int[lev_i]	= (hy_ai[lev_i] * 10**3.0 + hy_bi[lev_i] * pres)
			
		#Get to log (for interpolation)
		P_3_log		= np.log(P_3)
		P_3_int_log	= np.log(P_3_int)

		for lat_i in range(len(lat)):
			for lon_i in range(len(lon)):
				#Get the pressure coordinates for given point and interpolate
				x, y		= P_3_log[:, lat_i, lon_i], u_vel[:, lat_i, lon_i]
				x_int		= P_lev_log[P_lev_log <= x[-1]]
				u_vel_int	= interp1d(x,y)(x_int)
				
				x_v, y_v	= P_3_log[:, lat_i, lon_i], v_vel[:, lat_i, lon_i]
				x_int_v		= P_lev_log[P_lev_log <= x_v[-1]]
				v_vel_int	= interp1d(x_v,y_v)(x_int_v)
				
				x_w, y_w	= P_3_log[:, lat_i, lon_i], w_vel[:, lat_i, lon_i]
				x_int_w		= P_lev_log[P_lev_log <= x_w[-1]]
				w_vel_int	= interp1d(x_w,y_w)(x_int_w)
				
				#print(np.shape(u_vel_int))

				#Save velocity field
				u_vel_year[month_i-month_start, :len(u_vel_int), lat_i, lon_i]	= u_vel_int
				v_vel_year[month_i-month_start, :len(v_vel_int), lat_i, lon_i]	= v_vel_int
				w_vel_year[month_i-month_start, :len(w_vel_int), lat_i, lon_i]	= w_vel_int
				
				#print(month_i - month_start)
				
	#------------------------------------------------------------------------------
	month_days	= np.asarray([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31., 31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
	month_days	= month_days[month_start-1:month_end]

	#Fill the array's with the same dimensions
	month_days_all	= ma.masked_all((len(month_days), len(P_lev), len(lat), len(lon)))

	for month_i in range(len(month_days)):
		month_days_all[month_i]		= month_days[month_i]
		month_days_all[month_i]		= ma.masked_array(month_days_all[month_i], mask = u_vel_year[month_i].mask)

	month_days_all	= month_days_all / np.sum(month_days_all, axis = 0)

	#-----------------------------------------------------------------------------------------

	#Determine the time mean over the months of choice
	u_vel_year	= np.sum(u_vel_year * month_days_all, axis = 0)
	v_vel_year	= np.sum(v_vel_year * month_days_all, axis = 0)
	w_vel_year	= np.sum(w_vel_year * month_days_all, axis = 0)

	print(np.shape(u_vel_year))
	
	#Determine the weighted meridional mean
	lat_weight = lat_weight / np.sum(lat_weight)
	
	# Reshape lat_weight to (1, 6, 1) to match the shape of u_vel_year
	lat_weight = lat_weight[:, np.newaxis]
	
	print(np.shape(lat_weight))
	
	u_vel_all[year_i - int(np.min(time))]	= np.sum(u_vel_year * lat_weight, axis = 1)
	v_vel_all[year_i - int(np.min(time))]	= np.sum(v_vel_year * lat_weight, axis = 1)
	w_vel_all[year_i - int(np.min(time))]	= np.sum(w_vel_year * lat_weight, axis = 1)
	
	print(np.mean(u_vel_all[year_i - int(np.min(time))]))
	print(np.mean(np.mean(u_vel_year, axis=1)))
	
	#sys.exit()
	
	#Normalise the weighted area
	#lat_weight = lat_weight / np.sum(lat_weight)
	
	#Take weighted mean
	#for lev_i in range(len(P_lev)):
#		u_vel_mean[year_i - int(np.min(time)), lev_i]	= np.sum(lat_weight * u_vel_all[year_i - int(np.min(time)), lev_i, :])

#-----------------------------------------------------------------------------------------

print('Data is written to file')
fh = netcdf.Dataset(directory+'Atmosphere/VEL_meridional_mean_0S_5N_month_'+str(month_start)+'-'+str(month_end)+'_branch3800_year_'+str(year_start)+'_'+str(year_end)+'.nc', 'w')

fh.createDimension('time', len(time_year))
fh.createDimension('lev', len(P_lev))
fh.createDimension('lat', len(lat))
fh.createDimension('lon', len(lon))

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('lev', float, ('lev'), zlib=True)
fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('lon', float, ('lon'), zlib=True)
fh.createVariable('U', float, ('time', 'lev', 'lon'), zlib=True)
fh.createVariable('V', float, ('time', 'lev', 'lon'), zlib=True)
fh.createVariable('W', float, ('time', 'lev', 'lon'), zlib=True)

fh.variables['time'].long_name		= 'Time'
fh.variables['lev'].long_name		= 'Array of pressure levels'
fh.variables['lat'].long_name		= 'Array of latitudes used for meridional mean'
fh.variables['lon'].long_name		= 'Array of longitudes'
fh.variables['U'].long_name		= 'Time mean meridional mean zonal velocity'
fh.variables['V'].long_name		= 'Time mean meridional mean meridional velocity'
fh.variables['W'].long_name		= 'Time mean meridional mean vertical velocity'

fh.variables['time'].units 		= 'Model year'
fh.variables['lev'].units 		= 'hPa'
fh.variables['lat'].units 		= 'Degrees N'
fh.variables['lon'].units 		= 'Degrees E'
fh.variables['U'].units 		= 'm/s'
fh.variables['V'].units 		= 'm/s'
fh.variables['W'].units 		= 'Pa/s'

#Writing data to correct variable
fh.variables['time'][:] 		= time_year
fh.variables['lev'][:] 			= P_lev
fh.variables['lat'][:] 			= lat
fh.variables['lon'][:] 			= lon
fh.variables['U'][:] 			= u_vel_all
fh.variables['V'][:] 			= v_vel_all
fh.variables['W'][:] 			= w_vel_all

fh.close()
