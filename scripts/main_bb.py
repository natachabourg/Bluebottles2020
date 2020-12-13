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
date_summer_stings = [summer(d) for d in date_stings]

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











