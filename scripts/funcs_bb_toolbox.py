#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:47:22 2020

@author: natachab

Script storing all functions used 
in the new BB analysis

"""
import glob
import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm 
import matplotlib.colors as colors
import datetime
from datetime import timedelta
import xarray as xar
from decimal import Decimal, ROUND_HALF_UP
from windrose import plot_windrose
import seaborn as sns
from windrose import WindroseAxes
import cmocean as cmo

class time:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
        
    def jan_to_01(self):
        list_name=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        list_numbers=['1','2','3','4','5','6','7','8','9','10','11','12']
        for i in range(len(list_name)):
            if self.month==list_name[i]:
                self.month=list_numbers[i]
                


def GetVariables(filename,version):
    """
    Return date, water temp, #of bluebottles of a file
    """
    date, datee, water_temp, bluebottles, description = [], [], [], [], []
    for i in range(0,len(filename)):
        day=''
        month=''
        year=''
        if version=='old':
            date.append(str(filename.Name[i][:-12]))

            for j in range(0,2):
                if(date[i][j]!='/'):
                    day+=date[i][j]
            for j in range(2,len(date[i])-4):
                if(date[i][j]!='/'):
                    month+=date[i][j]
            for j in range(len(date[i])-4,len(date[i])):
                if(date[i][j]!='/'):
                    year+=date[i][j] 
        
        else:
            day = filename.Name[i].day
            month = filename.Name[i].month
            year = filename.Name[i].year
        
        if filename.Water_temp[i]!=14: #dont take values for water_temp=14C
            datee.append(time(str(day),str(month),str(year)))
            water_temp.append(filename.Water_temp[i])
            description.append(filename.Description[i])
            if filename.Bluebottles[i]=='none' or filename.Bluebottles[i]=='likely':
                bluebottles.append(0.)
            elif filename.Bluebottles[i]=='some':
                bluebottles.append(1.)
            elif filename.Bluebottles[i]=='many':
                bluebottles.append(2.)

    middle_date = []
    final_date, final_water_temp, final_bluebottles, final_description = [], [], [], []
    for l in range(len(datee)):
        middle_date.append(datetime.date(int(datee[l].year), int(datee[l].month), int(datee[l].day)))
    
    final_date.append(middle_date[0])
    final_water_temp.append(water_temp[0])
    final_bluebottles.append(bluebottles[0])
    final_description.append(description[0])
    
    for l in range(1,len(middle_date)):  
        if middle_date[l]!=middle_date[l-1]: #to only have one value per day
            final_date.append(middle_date[l])
            final_water_temp.append(water_temp[l])
            final_bluebottles.append(bluebottles[l])
            final_description.append(description[l])
            
    
    return final_date, final_water_temp, final_bluebottles, final_description



def choose_per_season(bluebottle_data, date_data, idx_beaching):
    """
    
    Input: 
        - bluebottle_data: array of bluebottles sightings 
        from the function gather_data() for a specific beach

        - date_data: date array from the function gather_data() for a 
        specific beach
        
        - idx_beaching: choose between 0=None, 1=Some, 2 = Many, [1,2]= Observed
    
    Output: nb of idx_beaching (0 for none, 1 for some, 2 for many, 1.5 for observed)
    observed per season per beach
    """
    
    bb_series = pd.Series(bluebottle_data)
    
    date = pd.Series(np.asarray(date_data).astype('datetime64'))
    
    month_data = bb_series.groupby(date.dt.month)
    
    nb = np.zeros(12)
    
    for i in range(1,13): #for each month
        
        if i in month_data.groups: 
            
            index = month_data.groups[i] 
            
            data = bb_series[index] #get the monthly data
            
            count = data.groupby(np.isin(data,idx_beaching)).count()
            
            if len(count)>1: #if no beachings no true array
                nb[i-1] = count[1]#/len(data)#compute % of beachings
            else:
                nb[i-1] = 0
            
        else: #happens when theres no data in the month considered 
            nb[i-1] = np.nan


    nb_djf = np.nansum([ nb[11], nb[0], nb[1] ])
    nb_mam = np.nansum([ nb[2], nb[3], nb[4] ])
    nb_jja = np.nansum([ nb[5], nb[6], nb[7] ])
    nb_son = np.nansum([ nb[8], nb[9], nb[10] ])


    return nb_djf, nb_mam, nb_jja, nb_son





def gather_data(path_obs, files_name_old, files_name_new):
    
    """
    function to gather bb observations data from all 
    lifeguard files and get variables needed
    
    Input : filename of both set of excel files
    Output: date, water_temp, bluebottle, description
    """
    
    #import files 
    
    beach_old = [pd.read_excel(f) for f in files_name_old]
    beach_new_temp = [pd.read_excel(f) for f in files_name_new]
    
    #to reverse the order of the file (went from 2020 to 2019)
    beach_new = [b.sort_index(ascending=False, axis=0) for b in beach_new_temp]
    for b in beach_new:
        b.index = b.index[::-1]
    
    #Get variables
    
    date_bb=[0,1,2]
    date=[0,1,2]
    water_temp=[0,1,2]
    bluebottles=[0,1,2]
    description=[0,1,2]
    
    date_bb_new=[0,1,2]
    date_new=[0,1,2]
    water_temp_new=[0,1,2]
    bluebottles_new=[0,1,2]
    description_new=[0,1,2]
    
    for i in range(0,len(files_name_old)):
        date_bb[i],water_temp[i], bluebottles[i], description[i] = GetVariables(beach_old[i],'old')
    
    for i in range(0, len(files_name_new)):
        date_bb_new[i],water_temp_new[i], bluebottles_new[i], description_new[i] = GetVariables(beach_new[i],'new')
    
    
    #delete data before 05/2016
    date[0]=date_bb[0]
    date[1]=date_bb[1][:1036] 
    date[2]=date_bb[2][:1025] 
    
    water_temp[1]=water_temp[1][:1036]
    water_temp[2]=water_temp[2][:1025] 
    
    bluebottles[1]=bluebottles[1][:1036]
    bluebottles[2]=bluebottles[2][:1025] 
    
    description[1]=description[1][:1036]
    description[2]=description[2][:1025] 
    
    #concatenate variables from old and new files
    
    for i in range(0,3):
        date[i] = np.concatenate([date[i], date_bb_new[i]])
        water_temp[i] = np.concatenate([water_temp[i], water_temp_new[i]])
        bluebottles[i] = np.concatenate([bluebottles[i], bluebottles_new[i]])
        description[i] = np.concatenate([description[i], description_new[i]])
        
    return date, water_temp, bluebottles, description



def get_kurnell_data_as_modified(folder):
    
    """
    
    To get Wind data from Kurnell from 2000 to 2020, 
    
    NB : computes daily averages from 9am to 9am local time
    
    and go from meteo to oceano convention
    
    Input : folder where the data is
    Output : dataframe with daily averaged:
        -upwell index
        -wind speed(m/s)
        -wind V (m/s)
        -wind U(m/s)
        -wind stress U 
        -wind stress V
        -air temp
        -RH
        -MSLP
        -wind speed (km/h)
        -wind gust speed (km/h)
        -precipitation (mm)
    
    """
    
    # PICK dataset: 
    file_name = 'HM01X_Data_066043_999999999777948' # KURNELL
    STATION = file_name[12:17]
    
    ### READ INFO station
    df_BOM_stations0 = pd.read_csv(folder + 'HM01X_StnDet_999999999777948.txt',header = None, dtype=object)#,usecols=[1,2,3,4,5,6,12,14,16,18,20,22,24,26])  # CAREFULL
    #df_BOM = df_BOM_stations0.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! 
    df_BOM_stations0
    
    # FIND STATION
    bbb = df_BOM_stations0[1].str.find(STATION)
    idx_station = bbb==1
    
    print(df_BOM_stations0[:][idx_station])
    
    STATION_isSYDNEY = df_BOM_stations0[3][idx_station].str.find("SYDNEY")
    print(np.array(STATION_isSYDNEY)==0) # 0 if includes SYDNEY, -1 otherwise... ??
    
    # For upwelling index, just a few
    rho_w = 1024 # sea water density kg/m3
    
    if (('Data_066043' in file_name) or (np.array(STATION_isSYDNEY)==0)): # Around Sydney
        lat = -33.9829
        coast_deg_angle = - 25      
    elif 'Data_092124' in file_name: #'MAI
        lat = -42.6621
        coast_deg_angle = - 15
    elif 'Data_040043' in file_name: #'NSI
        lat = -27.0314
        coast_deg_angle = - 350
    else:
        lat = 'np.nan'
        
    #print('Do you have the coastline angle to compute the upwelling index?')
    #~np.isnan(lat)
    
    
    ### READ DATA BOM
    df_BOM0 = pd.read_csv(folder + file_name + '.txt',usecols=[1,2,3,4,5,6,12,14,16,18,20,22,24,26])  # CAREFULL
    df_BOM = df_BOM0.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! 
    print(list(df_BOM))
    # Rename columns
    df_BOM.columns = ['station_nb', 'year','month','day','hour','minute','Precip_since9AM_mm','Air_temp','Rel_hum_perc','Wind_speed_kmh','Wind_dir_deg_meteo','Wind_gust_speed_kmh','MSLP_hPa','station_P_hPa']
    
    # Create a datetime component and make it the index in UTC

    """
    NB
    shift dates to daily average from 9am to 9am 
    
    e.g. 
    before : 01/02 09:00 to 02/02 8:00 
    after :  01/02 00:00 to 23:00
    
    """
    times_dates_local = [datetime.datetime(df_BOM.year[i], df_BOM.month[i], df_BOM.day[i], df_BOM.hour[i], df_BOM.minute[i], 0, 0) for i in range(len(df_BOM)) ]
    times_dates_shifted = [t-datetime.timedelta(hours=9) for t in times_dates_local]


    times_dates_UTC = fun_SYDdatetime2UTC(times_dates_local)
    # Add in df
    df_BOM.insert(1,"date_local", times_dates_local, True) 
    df_BOM.insert(1,"date_UTC", times_dates_UTC, True) 

    
    """
    NB 
    keep local date in index
    """
    df_BOM.index = pd.to_datetime(times_dates_shifted)
    
    df_BOM = df_BOM.drop_duplicates(keep ='first') # Remove dup[licates when daylight saving - keep first
    
    
    
    # Conversion speed direction from Meteo to Oceano
    Wind_u = fun_GetU_meteo(df_BOM.Wind_speed_kmh.values/3.6,df_BOM.Wind_dir_deg_meteo.values) 
    Wind_v = fun_GetV_meteo(df_BOM.Wind_speed_kmh.values/3.6,df_BOM.Wind_dir_deg_meteo.values)
    Wind_dir_deg = fun_Dir_oceanoFROMmeteo_deg(df_BOM.Wind_dir_deg_meteo.values)
    
    # [uwd,vwd]=pol2cart(Wind_speed_ms0, np.radians(Wind_dir_deg02)); #CHECK OK
    ###  Wind stress
    [speed, angle] = cart2pol(Wind_u,Wind_v);
    rho_air=1.22;      # density of air NEW after 1.3
    cd = (0.61 + 0.063*np.abs(speed))*1e-3;
    cd[np.abs(speed) < 6] = 1.1e-3;
    tau = cd * rho_air * np.abs(speed) * speed;
    Wind_tau_u = tau*np.cos(angle); 
    Wind_tau_v = tau*np.sin(angle);
    
    # Wind rose wind stress from oceano
    [rho,phi] = cart2pol(Wind_tau_u, Wind_tau_v)
    
    # Upwelling index
    
    if ~np.isnan(lat):
    
        #Wind_u_rot = np.cos(coast_deg_angle * np.pi / 180) * Wind_u_ms + np.sin(coast_deg_angle * np.pi / 180) * Wind_v_ms;  #  across-shelf 
        #Wind_v_rot = - np.sin(coast_deg_angle * np.pi / 180) * Wind_u_ms + np.cos(coast_deg_angle * np.pi / 180) * Wind_v_ms;  #  along -shelf 
        Wind_tau_u_rot = np.cos(coast_deg_angle * np.pi / 180) * Wind_tau_u + np.sin(coast_deg_angle * np.pi / 180) * Wind_tau_v;  #  across-shelf 
        Wind_tau_v_rot = - np.sin(coast_deg_angle * np.pi / 180) * Wind_tau_u + np.cos(coast_deg_angle * np.pi / 180) * Wind_tau_v;  #  along -shelf 
    
        UI = Wind_tau_v_rot / (rho_w * f_coriolis(lat)) # UNITS m2 /s
    
    # Create new dataframe with only selected columns (e.g. don't want to take the mean of direction!!!)
    df_BOM_oceano = df_BOM[['Precip_since9AM_mm','Air_temp','Rel_hum_perc','MSLP_hPa','Wind_speed_kmh','Wind_dir_deg_meteo','Wind_gust_speed_kmh']]
    # Add in df
    df_BOM_oceano.insert(1,"Wind_tau_u", Wind_tau_u)  # or df_BOM_all_daily['Wind_tau_v'] = Wind_tau_v ?
    df_BOM_oceano.insert(1,"Wind_tau_v", Wind_tau_v) 
    df_BOM_oceano.insert(1,"Wind_u_ms", Wind_u) 
    df_BOM_oceano.insert(1,"Wind_v_ms", Wind_v) 
    df_BOM_oceano.insert(1,"Wind_speed_ms", df_BOM.Wind_speed_kmh.values/3.6) 
    
    if ~np.isnan(lat):
        df_BOM_oceano.insert(1,"Upwell_Index", UI) 
    
    df_BOM_oceano = df_BOM_oceano.round(3)
    
    [rho,phi] = cart2pol(df_BOM_oceano.Wind_u_ms.values, df_BOM_oceano.Wind_v_ms.values)
    
    ### Daily averages
    # Average
    # df_BOM_oceano_daily = df_BOM_oceano.groupby(df_BOM_oceano.index.date).mean().round(3)
    # df_BOM_oceano_daily.index = pd.to_datetime(df_BOM_oceano_daily.index) #### CAREFUL!
    df_BOM_oceano_daily = df_BOM_oceano.resample('1D').mean().round(3)
    
    # But do not average direction in degrees!!!
    # AND do not average precipitation since 9AM!
    df_BOM_oceano_daily = df_BOM_oceano_daily.drop(columns=['Wind_dir_deg_meteo','Precip_since9AM_mm'])
    
    ### Daily precipitations: get max (since each value is cummulated from 9am)
    # df_temporary = df_BOM[['Precip_since9AM_mm']]
    # df_temporary_max = df_BOM.groupby(df_BOM.index.date).max()
    df_temporary_max = df_BOM.resample('1D').max()
    
    # add to dataframe
    df_BOM_oceano_daily['Precip_since9AM_mm_MAX'] = df_temporary_max.Precip_since9AM_mm 

    return df_BOM_oceano_daily


def dates_observed_none(bluebottles, dates_bb, dates_BOM):
    """
    Input : 
        - bluebottle and dates array from lifeguard reports  
        - dates array from BOM
    
    Outputs : 0:Clovelly, 1:Coogee, 2:Maroubra
        - dates of BB observed (some+many) and of BB none (none+likely), for each beach
        - index (adapted to BOM df data) when BB observed and BB none, for each beach 

    
    """
    dates_none = [[], [], []]
    dates_observed = [[], [], []]
    i_none = [[], [], []]
    i_observed = [[], [], []]
    
    dates_wind = np.array([pd.to_datetime(d) for d in dates_BOM])

    for i in range(0,len(bluebottles)):
        dates_beach = np.array([pd.to_datetime(d) for d in dates_bb[i]])
        
        dates_none[i] = dates_beach[np.where(bluebottles[i]==0)]

        dates_observed[i] = dates_beach[np.where(bluebottles[i]!=0)]
        
        i_none[i] = np.isin(dates_wind[:],dates_none[i])
        i_observed[i] = np.isin(dates_wind[:],dates_observed[i])
    
    return dates_none, dates_observed, i_none, i_observed




def summer(date):
    """
    Input: date array 
    Output: only summer (Austral:DJF) dates of the date array in input
    """
    months = np.array([getattr(d,'month') for d in date])
    date_sum = date[np.logical_or(months <= 2, months >= 12)]

    return date_sum


def winter(date):
    
    """
    Input: date array 
    Output: only winter (Austral:JJA) dates of the date array in input
    """
    
    months = np.array([getattr(d,'month') for d in date])
    date_win = date[np.logical_or(months <= 8, months >= 6)]

    return date_win

# Function for wind vectors (from AS)
def fun_GetU_meteo(speed,direction_inMETconv):
    wind_u = - speed * np.sin(np.pi / 180 * direction_inMETconv) # yes it's a - sin for the direction in met convention
    return wind_u
    
def fun_GetV_meteo(speed,direction_inMETconv):
    wind_v = - speed * np.cos(np.pi / 180 * direction_inMETconv)
    return wind_v

def fun_Dir_oceanoFROMmeteo_deg(Wind_dir_deg_meteo):
    Wind_dir_deg = (90 - Wind_dir_deg_meteo + 180);
    Wind_dir_deg[Wind_dir_deg <= 0] = Wind_dir_deg[Wind_dir_deg <= 0]+360;
    return Wind_dir_deg

def fun_Dir_meteoFROMoceano_deg(Wind_dir_deg_oceano):
    Wind_dir_deg = (90 - Wind_dir_deg_oceano + 180);
    Wind_dir_deg[Wind_dir_deg <= 0] = Wind_dir_deg[Wind_dir_deg <= 0]+360;
    return Wind_dir_deg

def fun_SYDtime2UTC(local_t):
    UTC_OFFSET = 10
    result_utc_time = local_t - UTC_OFFSET/24
    return result_utc_time

def fun_SYDdatetime2UTC(local_datetime):
    UTC_OFFSET = 10
    result_utc_datetime = [local_datetime[i] - datetime.timedelta(hours=UTC_OFFSET) for i in range(len(local_datetime)) ]
    return result_utc_datetime


def pol2cart(rho, phi):
    """
    author : Dr. Schaeffer
    """ 
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def cart2pol(x, y):
    """
    author : Dr. Schaeffer
    """   
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def rad2deg360(phi):
    """
    author : Dr. Schaeffer
    """ 
    x = np.degrees(phi)
    x[x < 0] = x[x < 0]+360;
    return(x)



# Function for upwell index
def f_coriolis(lat):  
    """Compute the Coriolis parameter for the given latitude:
    ``f = 2*omega*sin(lat)``, where omega is the angular velocity 
    of the Earth.
    
    Parameters
    ----------
    lat : array
      Latitude [degrees].
    """
    omega   = 7.2921159e-05  # angular velocity of the Earth [rad/s]
    return 2*omega*np.sin(lat/360.*2*np.pi)

# Functions to plot windrose

def fun_RosePlot(time_obs,direction_obs,speed_obs,title):
    ############# CARERFUL needs degrees between 0 and 360!!!!!!!!!!!!!!!!! use rad2deg360(phi)
    """
    returns a rose plot of the wind 
    """
    df = pd.DataFrame({"speed": speed_obs, "direction": direction_obs})
    bins = np.arange(0.01, 24, 4)
    fig=plt.figure()
    plot_windrose(df, kind="bar", normed=True, opening=0.8, nsector=8, edgecolor="white",bins=bins)
    plt.title('Wind blowing from, ' + title)


def fun_RosePlot_stress(time_obs,direction_obs,speed_obs,title):
    ############# CARERFUL needs degrees between 0 and 360!!!!!!!!!!!!!!!!! use rad2deg360(phi)
    """
    returns a rose plot of the wind stress
    """
    df = pd.DataFrame({"speed": speed_obs, "direction": direction_obs})
    bins = np.arange(0.01, 0.4, 0.05)
    fig=plt.figure()
    plot_windrose(df, kind="bar", normed=True, opening=0.8, nsector=8, edgecolor="white",bins=bins)
    plt.title('Wind stress blowing from, ' + title)


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



