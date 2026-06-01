#Program determines the velocities over the Southern Ocean

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
	lat 	= fh.variables['lat'][0:48]		#Latitude 
	pres	= fh.variables['PS'][0, 0:48,:] * 0.01	#Surface pressure (hPa)
	hy_am	= fh.variables['hyam'][:]		#Hybrid A coefficient at layer mid-point
	hy_bm	= fh.variables['hybm'][:]		#Hybrid B coefficient at layer mid-point
	hy_ai	= fh.variables['hyai'][:]		#Hybrid A coefficient at layer interface
	hy_bi	= fh.variables['hybi'][:]		#Hybrid B coefficient at layer interface
	U   	= fh.variables['U'][0, :, 0:48, :] 		#Zonal velocity (m/s)
	V 	= fh.variables['V'][0, :, 0:48, :]
	UU   	= fh.variables['UU'][0, :, 0:48, :] 		#Zonal velocity (m/s)
	VV 	= fh.variables['VV'][0, :, 0:48, :]
	lat_weight 	= fh.variables['gw'][0:48]

	fh.close()
	
	#print(np.shape(lat_weight))
	print(np.shape(lat))
	print(np.shape(lon))
	print(np.shape(pres))
	print(np.shape(hy_am))
	print(np.shape(hy_ai))
	print(U.shape)
	print(UU.shape)
	
	print(lat)
	print(lon)
	
	#sys.exit()

	#Convert the grid to -180 to 180 grid
#	lon_2, u_vel	= ConverterField3D(lon, u_vel)
#	lon_2, temp	= ConverterField3D(lon, temp)
#	lon, pres	= ConverterField2D(lon, pres)
	
	#print(lon)
	
	#Atlantic sector -80 to 15
	#lat_min_index	= (np.abs(lat - -2)).argmin()
	#lat_max_index	= (np.abs(lat - 2)).argmin()+1
#	lon_min_index	= (np.abs(lon - -80)).argmin()
#	lon_max_index	= (np.abs(lon - 15)).argmin()+1
	
#	lon		= lon[lon_min_index:lon_max_index]
	#lat		= lat[lat_min_index:lat_max_index]
	#lat_weight 	= lat_weight[lat_min_index:lat_max_index]
	#pres		= pres[lat_min_index:lat_max_index, lon_min_index:lon_max_index]
	#u_vel		= u_vel[:, lat_min_index:lat_max_index, lon_min_index:lon_max_index]
	#temp		= temp[:, lat_min_index:lat_max_index, lon_min_index:lon_max_index]
#	pres		= pres[:, lon_min_index:lon_max_index]
#	u_vel		= u_vel[:, :, lon_min_index:lon_max_index]
#	temp		= temp[:, :, lon_min_index:lon_max_index]
	#pres		= pres[lat_min_index:lat_max_index, :]
	#u_vel		= u_vel[:, lat_min_index:lat_max_index, :]
	#temp		= temp[:, lat_min_index:lat_max_index, :]

	return lon, lat, lat_weight, hy_am, hy_bm, hy_ai, hy_bi, pres, U, V, UU, VV

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

month_start	= 12
month_end	= 14

year_start	= 3299
year_end	= 3400

#-----------------------------------------------------------------------------------------

files = glob.glob(directory_data_2900+'*.cam2.h0.*.nc')
files.sort()

#print(len(files))

#print(files[0])
#print(files[-1])

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

#print(files[0])
#print(files[-1])

#print(time)

#sys.exit()

#-----------------------------------------------------------------------------------------

#Empty array for all the zonal means
lon, lat, lat_weight, hy_am, hy_bm, hy_ai, hy_bi, pres, U, V, UU, VV = ReadinData(files[0])

#Get the pressure levels for pre-defined levels
P_lev		= np.asarray([4, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 925, 950, 975, 1000])
P_lev_log	= np.log(P_lev)
time_year	= ma.masked_all(int(len(time)/12))
time_month	= time
u_vel_all	= ma.masked_all((len(time_month), len(lat), len(lon)))
v_vel_all	= ma.masked_all((len(time_month), len(lat), len(lon)))
uu_vel_all 	= ma.masked_all((len(time_month), len(lat), len(lon)))
vv_vel_all 	= ma.masked_all((len(time_month), len(lat), len(lon)))

for year_i in range(int(np.min(time)), int(np.min(time))+len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i - int(np.min(time))] = year_i
	
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

		lon, lat, lat_weight, hy_am, hy_bm, hy_ai, hy_bi, pres, U, V, UU, VV	= ReadinData(filename)

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
				x, y		= P_3_log[:, lat_i, lon_i], U[:, lat_i, lon_i]
				x_int		= P_lev_log[P_lev_log <= x[-1]]
				u_vel_int	= interp1d(x,y)(x_int)
				
				x, y		= P_3_log[:, lat_i, lon_i], V[:, lat_i, lon_i]
				x_int		= P_lev_log[P_lev_log <= x[-1]]
				v_vel_int 	= interp1d(x,y)(x_int)
				
				x, y		= P_3_log[:, lat_i, lon_i], UU[:, lat_i, lon_i]
				x_int		= P_lev_log[P_lev_log <= x[-1]]
				uu_vel_int	= interp1d(x,y)(x_int)
				
				x, y		= P_3_log[:, lat_i, lon_i], VV[:, lat_i, lon_i]
				x_int		= P_lev_log[P_lev_log <= x[-1]]
				vv_vel_int 	= interp1d(x,y)(x_int)
				
				#print(np.shape(u_vel_int))

				#Save velocity field
				u_vel_all[(year_i - int(np.min(time)))*12 + (month_i-month_start), lat_i, lon_i]    = u_vel_int[15]
				v_vel_all[(year_i - int(np.min(time)))*12 + (month_i-month_start), lat_i, lon_i]   = v_vel_int[15]			
				uu_vel_all[(year_i - int(np.min(time)))*12 + (month_i-month_start), lat_i, lon_i]  = uu_vel_int[15]
				vv_vel_all[(year_i - int(np.min(time)))*12 + (month_i-month_start), lat_i, lon_i]  = vv_vel_int[15]

		
#-----------------------------------------------------------------------------------------

print('Data is written to file')
fh = netcdf.Dataset(directory+'Atmosphere/Jet_200hPa_Southern_Ocean_month_'+str(month_start)+'-'+str(month_end)+'_branch2900_year_'+str(year_start)+'_'+str(year_end)+'.nc', 'w')

fh.createDimension('time', len(time_year))
fh.createDimension('time_month', len(time))
fh.createDimension('lev', len(P_lev))
fh.createDimension('lat', len(lat))
fh.createDimension('lon', len(lon))

fh.createVariable('time', float, ('time'), zlib=True)
fh.createVariable('time_month', float, ('time_month'), zlib=True)
fh.createVariable('lev', float, ('lev'), zlib=True)
fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('lon', float, ('lon'), zlib=True)
fh.createVariable('U', float, ('time_month', 'lat', 'lon'), zlib=True)
#fh.createVariable('U_month', float, ('time_month', 'lev', 'lat'), zlib=True)
#fh.createVariable('U_mean', float, ('time', 'lev'), zlib=True)
#fh.createVariable('TEMP', float, ('time', 'lev', 'lat'), zlib=True)
fh.createVariable('V', float, ('time_month', 'lat', 'lon'), zlib=True)
fh.createVariable('UU', float, ('time_month', 'lat', 'lon'), zlib=True)
fh.createVariable('VV', float, ('time_month', 'lat', 'lon'), zlib=True)

fh.variables['time'].long_name		= 'Time'
fh.variables['time_month'].long_name	= 'Monthly time'
fh.variables['lev'].long_name		= 'Array of pressure levels'
fh.variables['lat'].long_name		= 'Array of latitudes'
fh.variables['lon'].long_name		= 'Array of longitudes used for zonal mean'
fh.variables['U'].long_name		= 'Annual zonal mean zonal velocity'
#fh.variables['U_month'].long_name	= 'Monthly zonal mean zonal velocity'
#fh.variables['U_mean'].long_name	= 'Area weighted mean zonal velocity'
#fh.variables['TEMP'].long_name		= 'Zonal mean potential temperature'

fh.variables['time'].units 		= 'Model year'
fh.variables['time_month'].units 	= 'Time in months'
fh.variables['lev'].units 		= 'hPa'
fh.variables['lat'].units 		= 'Degrees N'
fh.variables['lon'].units 		= 'Degrees E'
fh.variables['U'].units 		= 'm/s'
#fh.variables['U_month'].units 		= 'm/s'
#fh.variables['U_mean'].units 		= 'm/s'
#fh.variables['TEMP'].units 		= 'K'

#Writing data to correct variable
fh.variables['time'][:] 		= time_year
fh.variables['time_month'][:] 		= time_month
fh.variables['lev'][:] 			= P_lev
fh.variables['lat'][:] 			= lat
fh.variables['lon'][:] 			= lon
fh.variables['U'][:] 			= u_vel_all
#fh.variables['U_month'][:] 		= u_vel_month
#fh.variables['U_mean'][:] 		= u_vel_mean
#fh.variables['TEMP'][:] 		= temp_all
fh.variables['V'][:]                    = v_vel_all
fh.variables['UU'][:]                    = uu_vel_all
fh.variables['VV'][:]                    = vv_vel_all

fh.close()
