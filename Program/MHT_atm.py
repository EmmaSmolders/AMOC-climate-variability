#Program determines the meridional heat transport

from pylab import *
import numpy
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

	fh = netcdf.Dataset(filename, mode='r')

	lon	= fh.variables['lon'][:]		#Longitudes (degrees E)
	lat	= fh.variables['lat'][:]		#Latitudes (degrees N)
	weight	= fh.variables['gw'][:]			#Gaussian weight for the zonal averages
	hy_ai	= fh.variables['hyai'][:]		#Hybrid A coefficient at layer interfaces
	hy_bi	= fh.variables['hybi'][:]		#Hybrid B coefficient at layer interfaces
	i_lev	= fh.variables['ilev'][:]		#Hybrid level at interfaces

	SW_TOA  = fh.variables['FSNT'][0]		#Net short wave af top of model (W/m^2)
	LW_TOA  = fh.variables['FLNT'][0]		#Net short wave af top of model (W/m^2)
	pres	= fh.variables['PS'][0]			#Surface pressure (Pa)
	v_vel	= fh.variables['V'][0]			#Meridional velocity (m/s)
	temp	= fh.variables['T'][0]			#Temperature (K)
	v_temp	= fh.variables['VT'][0]			#Meridional heat transport (K m/s)
	Q	= fh.variables['Q'][0]			#Specific humidity (kg/kg)			
	v_Q	= fh.variables['VQ'][0]			#Meridional water transport (m/s kg/kg)	
	Z_3	= fh.variables['Z3'][0]			#Geopotential height (m)

	fh.close()

	return lon, lat, weight, hy_ai, hy_bi, i_lev, SW_TOA, LW_TOA, pres, v_vel, temp, v_temp, Q, v_Q, Z_3

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------

year_start	= 4199
year_end	= 4300

#-----------------------------------------------------------------------------------------

files = glob.glob(directory_data_3800+'*.cam2.h0.*.nc')
files.sort()

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#Define empty array's
time 		= np.zeros(len(files))

for year_i in range(len(files)):
	date  = files[year_i][-10:-3]	
	year  = int(date[0:4])
	month = int(date[5:7])

	time[year_i] = year + (month-1) / 12.0

time_start	= (np.abs(time - year_start)).argmin()
time_end	= (np.abs(time - (year_end))).argmin()+12

time		= time[time_start:time_end]
files		= files[time_start:time_end]

print(files[0])
print(files[-1])

sys.exit()

#-----------------------------------------------------------------------------------------

#Empty array for all the zonal means
lon, lat, weight, hy_ai, hy_bi, i_lev, SW_TOA, LW_TOA, pres, v_vel, temp, v_temp, Q, v_Q, Z_3 = ReadinData(files[0])

#Get the pressure levels for pre-defined levels
P_lev		= np.asarray([4, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 925, 950, 975, 1000])
P_lev_log	= np.log(P_lev)

time_year	= ma.masked_all(int(len(time)/12))
MHF_all		= ma.masked_all((len(time_year), len(lat)))
SHF_all		= ma.masked_all((len(time_year), len(lat)))
SHF_eddy_all	= ma.masked_all((len(time_year), len(lat)))
SHF_stat_all	= ma.masked_all((len(time_year), len(lat)))
LHF_all		= ma.masked_all((len(time_year), len(lat)))
LHF_eddy_all	= ma.masked_all((len(time_year), len(lat)))
LHF_stat_all	= ma.masked_all((len(time_year), len(lat)))


for year_i in range(len(time_year)):
	#Now determine for each month
	print(year_i)
	time_year[year_i] = year_i + year_start
	files_month 	  = files[year_i*12:(year_i+1)*12]

	MHF_year	= ma.masked_all((12, len(lat)))
	SHF_year	= ma.masked_all((12, len(lat)))
	SHF_eddy_year	= ma.masked_all((12, len(lat)))
	SHF_stat_year	= ma.masked_all((12, len(lat)))
	LHF_year	= ma.masked_all((12, len(lat)))
	LHF_eddy_year	= ma.masked_all((12, len(lat)))
	LHF_stat_year	= ma.masked_all((12, len(lat)))

	for file_i in range(len(files_month)):
		print(files_month[file_i])
		lon, lat, weight, hy_ai, hy_bi, i_lev, SW_TOA, LW_TOA, pres, v_vel, temp, v_temp, Q, v_Q, Z_3	= ReadinData(files_month[file_i])

		#TOA difference
		heat_imbalance	= np.sum(SW_TOA - LW_TOA, axis = 1)
		MHF		= np.zeros(len(lat))

		for lat_i in range(1, len(lat)):
			MHF[lat_i]	= MHF[lat_i - 1] + ((6.371 * 10**6.0)**2.0 * 2 * np.pi / len(lon)) * heat_imbalance[lat_i] * weight[lat_i] / 10**15.0

		#Save monthly averaged value
		MHF_year[file_i]	= MHF

		#Meridional transport of geopotential
		v_Z	= v_vel * Z_3 * 9.81

		#Specific heat
		C_p	= (10**3.0) * (1.005 + (1.82 * Q))

		#Latent heat
		L_v	= (10**3.0) * (2.5008e3 + (2.36 * (temp - 273.15)) + (0.0016 * (temp - 273.15)**2.0)+(0.00006 * (temp - 273.15)**3.0))

		#Determine the eddy components
		temp		= temp * C_p
		v_temp		= v_temp * C_p
		Q		= Q * L_v
		v_Q		= v_Q * L_v

		v_temp_eddy	= v_temp - (v_vel * temp)
		v_Q_eddy	= v_Q - (v_vel * Q)

		#Determine the stationary components, take the zonal means
		v_temp_stat	= np.mean(v_temp, axis = 2) - (np.mean(v_vel, axis = 2) * np.mean(temp, axis = 2)) - np.mean(v_temp_eddy, axis = 2)
		v_Q_stat	= np.mean(v_Q, axis = 2) - (np.mean(v_vel, axis = 2) * np.mean(Q, axis = 2)) - np.mean(v_Q_eddy, axis = 2)
		v_Z_stat	= np.mean(v_Z, axis = 2) - (9.81 * np.mean(v_vel, axis = 2) * np.mean(Z_3, axis = 2))

		#Hybrid sigma levels
		P_3	= np.zeros((len(i_lev), len(lat), len(lon)))

		for lev_i in range(len(i_lev)):
			#Determine the sigma levels using the pressure at the surface
			P_3[lev_i]	= hy_ai[lev_i] * 10**5.0 + hy_bi[lev_i] * pres

		#Determine the total mass inside a layer and the total vertical sum
		P_3_diff		= (1.0 / 9.81) * (P_3[1:] - P_3[:-1])
		v_temp_total		= np.sum(v_temp * P_3_diff, axis = 0)
		v_temp_eddy_total	= np.sum(v_temp_eddy * P_3_diff, axis = 0)
		v_Q_total		= np.sum(v_Q * P_3_diff, axis = 0)
		v_Q_eddy_total		= np.sum(v_Q_eddy * P_3_diff, axis = 0)
		v_Z_total		= np.sum(v_Z * P_3_diff, axis = 0) 

		#Determine the zonal mean, take into account the radius at each latitude
		radius_lat		= (2.0 * np.pi * (6.371 * 10**6.0) * np.cos(lat * np.pi / 180.0))
		SHF_year[file_i]	= np.mean(v_temp_total, axis = 1) * radius_lat
		SHF_eddy_year[file_i]	= np.mean(v_temp_eddy_total, axis = 1) * radius_lat
		LHF_year[file_i]	= np.mean(v_Q_total, axis = 1) * radius_lat
		LHF_eddy_year[file_i]	= np.mean(v_Q_eddy_total, axis = 1) * radius_lat

		#Determine the vertical integral of the stationary components, take into account the radius at each latitude
		P_3_lon			= np.mean(P_3, axis = 2)
		P_3_diff		= (1.0 / 9.81) * (P_3_lon[1:] - P_3_lon[:-1])
		SHF_stat_year[file_i]	= np.sum(v_temp_stat * P_3_diff, axis = 0) * radius_lat
		LHF_stat_year[file_i]	= np.sum(v_Q_stat * P_3_diff, axis = 0) * radius_lat
	
	#------------------------------------------------------------------------------
	month_days	= np.asarray([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
	month_days	= month_days / np.sum(month_days)

	#Fill the array's with the same dimensions
	month_days_all	= ma.masked_all((len(month_days), len(lat)))

	for month_i in range(len(month_days)):
		month_days_all[month_i]		= month_days[month_i]

	#-----------------------------------------------------------------------------------------
	#Determine the time mean over the months of choice
	MHF_all[year_i]		= np.sum(MHF_year * month_days_all, axis = 0)
	SHF_all[year_i]		= np.sum(SHF_year * month_days_all, axis = 0)
	SHF_eddy_all[year_i]	= np.sum(SHF_eddy_year * month_days_all, axis = 0)
	SHF_stat_all[year_i]	= np.sum(SHF_stat_year * month_days_all, axis = 0)
	LHF_all[year_i]		= np.sum(LHF_year * month_days_all, axis = 0)
	LHF_eddy_all[year_i]	= np.sum(LHF_eddy_year * month_days_all, axis = 0)
	LHF_stat_all[year_i]	= np.sum(LHF_stat_year * month_days_all, axis = 0)

#-----------------------------------------------------------------------------------------
print('Data is written to file')
fh = netcdf.Dataset(directory+'Atmosphere/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_3800.nc', 'w')

fh.createDimension('lat', len(lat))

fh.createVariable('lat', float, ('lat'), zlib=True)
fh.createVariable('MHT', float, ('lat'), zlib=True)
fh.createVariable('SHF', float, ('lat'), zlib=True)
fh.createVariable('SHF_eddy', float, ('lat'), zlib=True)
fh.createVariable('SHF_stat', float, ('lat'), zlib=True)
fh.createVariable('LHF', float, ('lat'), zlib=True)
fh.createVariable('LHF_eddy', float, ('lat'), zlib=True)
fh.createVariable('LHF_stat', float, ('lat'), zlib=True)

fh.variables['lat'].longname		= 'Array of latitudes'
fh.variables['MHT'].longname		= 'Meridional heat transport'
fh.variables['SHF'].longname		= 'Sensible heat flux'
fh.variables['SHF_eddy'].longname	= 'Sensible heat flux, eddy component'
fh.variables['SHF_stat'].longname	= 'Sensible heat flux, stationary component'
fh.variables['LHF'].longname		= 'Latent heat flux'
fh.variables['LHF_eddy'].longname	= 'Latent heat flux, eddy component'
fh.variables['LHF_stat'].longname	= 'Latent heat flux, stationary component'

fh.variables['lat'].units 	= 'degrees N'
fh.variables['MHT'].units 	= 'PW'
fh.variables['SHF'].units 	= 'PW'
fh.variables['SHF_eddy'].units 	= 'PW'
fh.variables['SHF_stat'].units 	= 'PW'
fh.variables['LHF'].units 	= 'PW'
fh.variables['LHF_eddy'].units 	= 'PW'
fh.variables['LHF_stat'].units 	= 'PW'

#Writing data to correct variable	
fh.variables['lat'][:]     	= lat
fh.variables['MHT'][:]     	= np.mean(MHF_all, axis = 0)
fh.variables['SHF'][:]     	= np.mean(SHF_all, axis = 0) / 10**15.0
fh.variables['SHF_eddy'][:]     = np.mean(SHF_eddy_all, axis = 0) / 10**15.0
fh.variables['SHF_stat'][:]     = np.mean(SHF_stat_all, axis = 0) / 10**15.0
fh.variables['LHF'][:]     	= np.mean(LHF_all, axis = 0) / 10**15.0
fh.variables['LHF_eddy'][:]     = np.mean(LHF_eddy_all, axis = 0) / 10**15.0
fh.variables['LHF_stat'][:]     = np.mean(LHF_stat_all, axis = 0) / 10**15.0

fh.close()

