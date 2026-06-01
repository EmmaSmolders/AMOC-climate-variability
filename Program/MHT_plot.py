#Program computes the meridional heat flux

#%%
from pylab import *
import numpy
import datetime
import time
import glob, os
import math
import netCDF4 as netcdf
import matplotlib.colors as colors
from scipy.interpolate import interp1d

#Making pathway to folder with all data
directory           = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Output/'
directory_data	    = '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Data/Atmosphere/'
directory_figures	= '/Users/6008399/Documents/PhD/2025/Projects/EWS_atmosphere/Figures/'

def ReadinData(filename):

	fh = netcdf.Dataset(filename, 'r')

	lat	= fh.variables['lat'][:]		#Latitudes (degrees N)
	MHT	= fh.variables['MHT'][:]		#Meridional heat transport (PW)

	fh.close()

	return lat, MHT

#-----------------------------------------------------------------------------------------
#--------------------------------MAIN SCRIPT STARTS HERE----------------------------------
#-----------------------------------------------------------------------------------------	

year_start		= 999
year_end		= 1100
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_on	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_600.nc')
lat_ocn, MHT_ocn_on	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_600.nc')

#-----------------------------------------------------------------------------------------

year_start		= 4199
year_end		= 4300
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_off	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_3800.nc')
lat_ocn, MHT_ocn_off	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_3800.nc')

#-----------------------------------------------------------------------------------------
year_start		= 1899
year_end		= 2000
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_on_45	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_1500.nc')
lat_ocn, MHT_ocn_on_45	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_1500.nc')

#-----------------------------------------------------------------------------------------

year_start		= 3299
year_end		= 3400
#-----------------------------------------------------------------------------------------

#Get the total and oceanic heat transport
lat_atm, MHT_tot_off_45	= ReadinData(directory_data+'/Meridional_heat_transport_atm_year_'+str(year_start)+'-'+str(year_end)+'_branch_2900.nc')
lat_ocn, MHT_ocn_off_45	= ReadinData(directory_data+'/Meridional_heat_transport_year_'+str(year_start)+'-'+str(year_end)+'_branch_2900.nc')

plt.figure()
plt.plot(lat_ocn, MHT_ocn_off, label='off 18')
plt.plot(lat_ocn, MHT_ocn_on, label='on 18')
plt.plot(lat_ocn, MHT_ocn_off_45, label = 'off 45')
plt.plot(lat_ocn, MHT_ocn_on_45, label = 'on 45')
plt.title('ocean')
plt.legend()

#%%
#Interpolate the oceanic heat transport to the atmospheric grid
#Skip the first 8 as there is no oceanic heat transport on Antarctic (of course)
MHT_ocn_on_int		= interp1d(lat_ocn, MHT_ocn_on)(np.array(lat_atm[8:]))
MHT_ocn_off_int		= interp1d(lat_ocn, MHT_ocn_off)(np.array(lat_atm[8:]))

MHT_atm_on	= np.copy(MHT_tot_on)
MHT_atm_off	= np.copy(MHT_tot_off)
MHT_atm_on[8:]  -= MHT_ocn_on_int
MHT_atm_off[8:] -= MHT_ocn_off_int
#-----------------------------------------------------------------------------------------

MHT_ocn_on_int_45		= interp1d(lat_ocn, MHT_ocn_on_45)(np.array(lat_atm[8:]))
MHT_ocn_off_int_45		= interp1d(lat_ocn, MHT_ocn_off_45)(np.array(lat_atm[8:]))

MHT_atm_on_45	= np.copy(MHT_tot_on_45)
MHT_atm_off_45	= np.copy(MHT_tot_off_45)
MHT_atm_on_45[8:]  -= MHT_ocn_on_int_45
MHT_atm_off_45[8:] -= MHT_ocn_off_int_45

plt.figure()
plt.plot(lat_ocn, MHT_ocn_off, label='off 18')
plt.plot(lat_ocn, MHT_ocn_on, label='on 18')
plt.plot(lat_ocn, MHT_ocn_off_45, label = 'off 45')
plt.plot(lat_ocn, MHT_ocn_on_45, label = 'on 45')
plt.title('ocean')
plt.legend()

plt.figure()
plt.plot(lat_atm, MHT_atm_off, label='off 18')
plt.plot(lat_atm, MHT_atm_on, label='on 18')
plt.plot(lat_atm, MHT_atm_off_45, label = 'off 45')
plt.plot(lat_atm, MHT_atm_on_45, label = 'on 45')
plt.title('atmosphere')
plt.legend()

#%%

plt.figure()
plt.plot(lat_atm, MHT_atm_off_45 - MHT_atm_on_45)

plt.figure()
plt.plot(lat_ocn, MHT_ocn_off_45 - MHT_ocn_on_45)

#%%
fig, ax	= subplots()

graph_1		= ax.plot(lat_atm, MHT_tot_off - MHT_tot_on, '-', color = 'k', linewidth = 2, label = 'Total')
graph_2		= ax.plot(lat_ocn, MHT_ocn_off - MHT_ocn_on, '-', color = 'royalblue', linewidth = 2, label = 'Ocean')
graph_3		= ax.plot(lat_atm, MHT_atm_off - MHT_atm_on, '-', color = 'firebrick', linewidth = 2, label = 'Atmosphere')


ax.set_ylabel('Meridional heat transport difference (PW)')
ax.set_xlim(-90, 90)
ax.set_ylim(-0.8, 0.8)
ax.grid()

ax.set_xticks(np.arange(-90, 90.1, 30))
ax.set_xticklabels(['90$^{\circ}$S', '60$^{\circ}$S', '30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])

graphs	      	= graph_1 + graph_2 + graph_3

legend_labels 	= [l.get_label() for l in graphs]
legend_2	= ax.legend(graphs, legend_labels, loc = 'upper left', ncol=1, framealpha = 1.0)


ax.set_title('a) Meridional heat transport PI$_{18}$')

show()

#-----------------------------------------------------------------------------------------

fig, ax	= subplots()

graph_1		= ax.plot(lat_atm, MHT_tot_off_45 - MHT_tot_on_45, '-', color = 'k', linewidth = 2, label = 'Total')
graph_2		= ax.plot(lat_ocn, MHT_ocn_off_45 - MHT_ocn_on_45, '-', color = 'royalblue', linewidth = 2, label = 'Ocean')
graph_3		= ax.plot(lat_atm, MHT_atm_off_45 - MHT_atm_on_45, '-', color = 'firebrick', linewidth = 2, label = 'Atmosphere')


ax.set_ylabel('Meridional heat transport difference (PW)')
ax.set_xlim(-90, 90)
ax.set_ylim(-0.8, 0.8)
ax.grid()

ax.set_xticks(np.arange(-90, 90.1, 30))
ax.set_xticklabels(['90$^{\circ}$S', '60$^{\circ}$S', '30$^{\circ}$S', '0$^{\circ}$', '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])

graphs	      	= graph_1 + graph_2 + graph_3

legend_labels 	= [l.get_label() for l in graphs]
legend_2	= ax.legend(graphs, legend_labels, loc = 'upper left', ncol=1, framealpha = 1.0)


ax.set_title('b) Meridional heat transport PI$_{45}$')

show()

#%% Print MHT at 26N for the different simulations

lat_atm_26N_index = np.abs(lat_atm - 26).argmin()
lat_ocn_26N_index = np.abs(lat_ocn - 26).argmin()

print(lat_atm[lat_atm_26N_index])
print(lat_ocn[lat_ocn_26N_index])

print("MHT at 26N for PI_18 on:", MHT_tot_on[lat_atm_26N_index], "PW")
print("MHT at 26N for PI_18 off:", MHT_tot_off[lat_atm_26N_index], "PW")
print("MHT at 26N for PI_45 on:", MHT_tot_on_45[lat_atm_26N_index], "PW")
print("MHT at 26N for PI_45 off:", MHT_tot_off_45[lat_atm_26N_index], "PW")
print("Oceanic MHT at 26N for PI_18 on:", MHT_ocn_on[lat_ocn_26N_index], "PW")
print("Oceanic MHT at 26N for PI_18 off:", MHT_ocn_off[lat_ocn_26N_index], "PW")
print("Oceanic MHT at 26N for PI_45 on:", MHT_ocn_on_45[lat_ocn_26N_index], "PW")
print("Oceanic MHT at 26N for PI_45 off:", MHT_ocn_off_45[lat_ocn_26N_index], "PW")
print("Atmospheric MHT at 26N for PI_18 on:", MHT_atm_on[lat_atm_26N_index], "PW")
print("Atmospheric MHT at 26N for PI_18 off:", MHT_atm_off[lat_atm_26N_index], "PW")
print("Atmospheric MHT at 26N for PI_45 on:", MHT_atm_on_45[lat_atm_26N_index], "PW")
print("Atmospheric MHT at 26N for PI_45 off:", MHT_atm_off_45[lat_atm_26N_index], "PW")

#%%

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# =======================
# Panel (a) PI_18
# =======================
axs[0].plot(lat_atm, MHT_tot_off - MHT_tot_on,
            color='k', linewidth=2, label='Total')
axs[0].plot(lat_ocn, MHT_ocn_off - MHT_ocn_on,
            color='royalblue', linewidth=2, label='Ocean')
axs[0].plot(lat_atm, MHT_atm_off - MHT_atm_on,
            color='firebrick', linewidth=2, label='Atmosphere')

axs[0].set_ylabel('MHT difference (PW)')
axs[0].set_xlim(-90, 90)
axs[0].set_ylim(-0.8, 0.8)
axs[0].set_title('a) Meridional heat transport (PI$_{18}$)')
axs[0].grid()
axs[0].legend(loc='upper left', framealpha=1.0)


# =======================
# Panel (b) PI_45
# =======================
axs[1].plot(lat_atm, MHT_tot_off_45 - MHT_tot_on_45,
            color='k', linewidth=2, label='Total')
axs[1].plot(lat_ocn, MHT_ocn_off_45 - MHT_ocn_on_45,
            color='royalblue', linewidth=2, label='Ocean')
axs[1].plot(lat_atm, MHT_atm_off_45 - MHT_atm_on_45,
            color='firebrick', linewidth=2, label='Atmosphere')

axs[1].set_ylabel('MHT difference (PW)')
axs[1].set_ylim(-0.8, 0.8)
axs[1].set_title('b) Meridional heat transport (PI$_{45}$)')
axs[1].grid()


# =======================
# Shared x-axis formatting
# =======================
axs[1].set_xticks(np.arange(-90, 90.1, 30))
axs[1].set_xticklabels(['90$^{\circ}$S', '60$^{\circ}$S', '30$^{\circ}$S',
                        '0$^{\circ}$',
                        '30$^{\circ}$N', '60$^{\circ}$N', '90$^{\circ}$N'])
axs[1].set_xlabel('Latitude')

plt.tight_layout()
plt.show()



# %%

plt.figure()
plt.plot(lat_atm, MHT_atm_off - MHT_atm_on, label='PI_18')
plt.plot(lat_atm, MHT_atm_off_45 - MHT_atm_on_45, label='PI_45')
plt.title('Atmospheric MHT difference')
plt.legend()    

plt.figure()
plt.plot(lat_ocn, MHT_ocn_off - MHT_ocn_on, label='PI_18')
plt.plot(lat_ocn, MHT_ocn_off_45 - MHT_ocn_on_45, label='PI_45')
plt.title('Oceanic MHT difference')
plt.legend()    

plt.figure()
plt.plot(lat_atm, MHT_tot_off - MHT_tot_on, label='PI_18')
plt.plot(lat_atm, MHT_tot_off_45 - MHT_tot_on_45, label='PI_45')
plt.title('Total MHT difference')
plt.legend()    
# %%
