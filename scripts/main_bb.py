#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:48:52 2020

@author: natachab

"""
import os
os.system('python func_bb_toolbox.py')
os.system('python func_bb_analysis.py')

from funcs_bb_toolbox import *
from funcs_bb_analysis import *


#%% import files  define paths
path_figs = '../figs/'
path_obs = '../data/'
files_name_old = [glob.glob(path_obs+'bluebottle_lifeguard_reports/Clov*2.xlsx')[0],
                  glob.glob(path_obs+'bluebottle_lifeguard_reports/Coo*2.xlsx')[0],
                  glob.glob(path_obs+'bluebottle_lifeguard_reports/Mar*2.xlsx')[0]]
                  
files_name_new = [glob.glob(path_obs+'bluebottle_lifeguard_reports/*Clov*3.xlsx')[0],
                  glob.glob(path_obs+'bluebottle_lifeguard_reports/*Coo*3.xlsx')[0],
                  glob.glob(path_obs+'bluebottle_lifeguard_reports/*Mar*3.xlsx')[0]]
  
                   
#gather old and new lifeguard reports
date, water_temp, bluebottles, description = gather_data(path_obs, 
                                                         files_name_old, 
                                                         files_name_new)

#Import wind data from Kurnell from 2000 to 2020 
df_BOM = get_kurnell_data_as_modified(folder = path_obs+'PUG_05_2020_AS_wind_Kurnell/DATA/')
U = df_BOM['Wind_u_ms']
V = df_BOM['Wind_v_ms']
date_BOM = df_BOM.index
ws, wd_oceano_rad = cart2pol(U,V)
wd_oceano_deg = rad2deg360(wd_oceano_rad)
wd_meteo_deg = fun_Dir_meteoFROMoceano_deg(wd_oceano_deg)

#Compute observed and non observed beaching days and get their indexes for BOM df
[date_none, 
date_observed, 
i_none, 
i_observed] = dates_observed_none(bluebottles, date, date_BOM)


# Get dates and bluebottle observations values for summer only
date_summer = [ date[i][np.isin(date[i], summer(date[i]))] for i in range(0,3)]
bb_summer = [ bluebottles[i][np.isin(date[i], summer(date[i]))] for i in range(0,3)]

#Read stings data
file_stings = pd.read_excel(path_obs+'bluebottle_lifeguard_reports/bluebottle_database_daniel.xlsx',
                            sheet_name = 'Sting Reports')
date_stings = [file_stings.Date[file_stings.Clovelly>0],
               file_stings.Date[file_stings['Coogee (NSW)']>0],
               file_stings.Date[file_stings.Maroubra>0]]

date_nostings = [file_stings.Date[file_stings.Clovelly==0],
               file_stings.Date[file_stings['Coogee (NSW)']==0],
               file_stings.Date[file_stings.Maroubra==0]]

date_summer_stings = [summer(d) for d in date_stings]
date_summer_nostings = [summer(d) for d in date_nostings]


#%% start seasonality  analysis

#BARPLOT OF BEACHING OCCURRENCES DEPENDING ON THE SEASON FOR ALL BEACHES
season_diff_between_beaches(bluebottles, 
                            date, 
                            idx_beaching = [1,2], 
                            path_fig = "../figs/")


#PLOT OF THE SEASONAL CYCLES OF BEACHINGS OVER THAT 
# OF SST, WIND
seasonal_cycle_plot(2,"../figs/",
                    date_none,date_observed,
                    date,bluebottles,water_temp,
                    date_BOM,df_BOM,
                    ws, wd_meteo_deg)

#get correlation coefficient between bluebottles and 
#sst, wind (speed, direction,across velocity,along velocity)
corr=correlation_btwn_variables(2,"../figs/",
                               date, bluebottles, water_temp,
                               date_BOM, df_BOM,
                               ws, wd_meteo_deg)

#%% start small timescale analysis 


#PLOT MAP WITH WIND CONDITIONS FOR BEACHING 
summer_rose_plot_map(date_summer_stings,date_summer,bb_summer,date_BOM,ws,wd_meteo_deg,
                     folder="../figs/",path_data="../data/")


#TABLE CHANCES OF BEACHING PER WIND DIRECTION CATEGORY AND FOR EACH BEACH
#FOR ALL DATES TOGETHER
chances = np.array([ chances_of_beachings_wd(wd_meteo_deg, 
                                      date_BOM, 
                                      date_none[beach], 
                                      date_observed[beach], 
                                      beach) for beach in range(0,3)])
#0 Clovelly 1 Coogee 2 Maroubra
ch_NE = chances[:,0]
ch_SE = chances[:,1]
ch_SW = chances[:,2]
ch_NW = chances[:,3]


#Get dates of beaching/no beaching dates only for summer
[date_none_summer, 
date_observed_summer,
i1,
i1] = dates_observed_none(np.array(bb_summer), np.array(date_summer), date_BOM)

del i1

#TABLE CHANCES OF BEACHING PER WIND DIRECTION CATEGORY AND FOR EACH BEACH
#ONLY FOR SUMMER
chances_summer = np.array([ chances_of_beachings_wd(wd_meteo_deg, 
                                      date_BOM, 
                                      date_none_summer[beach], 
                                      date_observed_summer[beach], 
                                      beach) for beach in range(0,3)])


#1 Coogee 2 Maroubra (No Clovelly)
ch_NE_s = chances_summer[:,0]
ch_SE_s = chances_summer[:,1]
ch_SW_s = chances_summer[:,2]
ch_NW_s = chances_summer[:,3]


#%% moon phase analysis

location=['Clovelly','Coogee','Maroubra']

for beach in range(0,3):
    phase = []
    for d in date[beach]:
        mi = pylunar.MoonInfo((-33, 51, 55), (151, 12, 36))
        mi.update((d.year,d.month,d.day,12,0))
        phase.append(mi.fractional_phase())
    phase = np.array(phase)
    
    date_phase=date[beach]
    ph = []
    ph.append(date_phase[phase<=0.1])
    ph.append(date_phase[np.logical_and(phase>0.1,phase<=0.2)])
    ph.append(date_phase[np.logical_and(phase>0.2,phase<=0.3)])
    ph.append(date_phase[np.logical_and(phase>0.3,phase<=0.4)])
    ph.append(date_phase[np.logical_and(phase>0.4,phase<=0.5)])
    ph.append(date_phase[np.logical_and(phase>0.5,phase<=0.6)])
    ph.append(date_phase[np.logical_and(phase>0.6,phase<=0.7)])
    ph.append(date_phase[np.logical_and(phase>0.7,phase<=0.8)])
    ph.append(date_phase[np.logical_and(phase>0.8,phase<=0.9)])
    ph.append(date_phase[np.logical_and(phase>0.9,phase<=1)])
    
    ph = np.array(ph)
    per = np.zeros(len(ph))
    for i in range(0,len(ph)):
        per[i]=len(ph[i][np.isin(ph[i],date_observed[beach])])/len(ph[i])
        
    x = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    
    fig,ax=plt.subplots()
    ax.set_xticks(np.arange(0,1,0.1))
    ax.set_xlabel('Fractional phase (0 = New Moon, 1=Full Moon)')
    ax.set_ylabel('Chances of beaching events (Some/All)')
        
    ax.scatter(x,per,c='b')
    ax.plot([0,0.1],[per[0],per[0]],c='b')
    ax.plot([0.1,0.2],[per[1],per[1]],c='b')
    ax.plot([0.2,0.3],[per[2],per[2]],c='b')
    ax.plot([0.3,0.4],[per[3],per[3]],c='b')
    ax.plot([0.4,0.5],[per[4],per[4]],c='b')
    ax.plot([0.5,0.6],[per[5],per[5]],c='b')
    ax.plot([0.6,0.7],[per[6],per[6]],c='b')
    ax.plot([0.7,0.8],[per[7],per[7]],c='b')
    ax.plot([0.8,0.9],[per[8],per[8]],c='b')
    ax.plot([0.9,1],[per[9],per[9]],c='b')
    ax.grid(zorder=0,c='lightgrey')
    ax.set_title(location[beach])
    fig.savefig('../figs/frequency_moon_'+str(location[beach])+'.png',dpi=400)
   
fig,ax=plt.subplots()
ax.grid(c='lightgrey',alpha=0.8,zorder=0)

ax.plot(date[beach],phase,zorder=1)
ax.scatter(date[beach],phase,label='none',zorder=2)
ax.scatter(date[beach][np.isin(date[beach],date_observed[beach])],phase[np.isin(date[beach],date_observed[beach])],
           c='red',label='observed',zorder=3)
ax.legend()
ax.set_title(location[beach])
ax.set_ylabel('Fractional phase (0 = New Moon, 1=Full Moon)')




date_test = np.array([datetime.datetime(2012,1,1)+datetime.timedelta(days=i) for i in range(0,100)])

phase_test = []
for d in date_test:
    mi = pylunar.MoonInfo((-33, 51, 55), (151, 12, 36))
    mi.update((d.year,d.month,d.day,12,0))
    phase_test.append(mi.fractional_phase())
phase_test = np.array(phase_test)
    
fig,ax = plt.subplots()
ax.plot(date_test,phase_test,label='fractional phase')
ax2=ax.twinx()
ax.grid(axis='x')
ax2.plot(date_test,np.gradient(phase_test),c='red',label='gradient(fractional phase)')
ax2.grid()
ax.set_ylabel('Fractional phase, Blue')
ax2.set_ylabel('Gradient(fractional phase), Red')
fig.tight_layout()

