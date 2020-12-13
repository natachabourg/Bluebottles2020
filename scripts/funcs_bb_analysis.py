#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:13:47 2020

@author: natachab
"""

from funcs_bb_toolbox import *


##---------------- SEASONALITY PLOTS ---------------#

def season_diff_between_beaches(bluebottles, date, idx_beaching, path_fig):
    
    """
    Input : 
        - bluebottles = Bluebottle sighting array from gather_data()
        - date = date array from gather_data()
        - idx_beaching = choice between  0=None, 1=Some, 2 = Many, [1,2]= Observed
        - path_fig = path where to save the fig
        
    Output:  Plot, for all beaches, the percentage of beachings events per season
    
    """
    
    #Get number of occurrence of each season
    nb_season = [choose_per_season(bluebottles[i], date[i],idx_beaching) 
                for i in range(0,len(bluebottles))]

    djf = [n[0] for n in nb_season]
    mam = [n[1] for n in nb_season]
    jja = [n[2] for n in nb_season]
    son = [n[3] for n in nb_season]
    
    #Get percentage of each season    
    per_djf = [ nb_season[i][0]/np.sum([nb_season[i][0],nb_season[i][1],nb_season[i][2],nb_season[i][3]]) for i in range(0,3)]
    per_mam = [ nb_season[i][1]/np.sum([nb_season[i][0],nb_season[i][1],nb_season[i][2],nb_season[i][3]]) for i in range(0,3)]
    per_jja = [ nb_season[i][2]/np.sum([nb_season[i][0],nb_season[i][1],nb_season[i][2],nb_season[i][3]]) for i in range(0,3)]
    per_son = [ nb_season[i][3]/np.sum([nb_season[i][0],nb_season[i][1],nb_season[i][2],nb_season[i][3]]) for i in range(0,3)]
    
    #To put bars on top of another
    x= [1,2,3]
    bars_1 = np.add(son, djf).tolist()
    bars_2 = np.add(bars_1,mam).tolist()
        
    #Choose colors
    a = pylab.cm.YlGnBu(.3)
    b = pylab.cm.YlGnBu(.4)
    c = pylab.cm.YlGnBu(.7)
    d = pylab.cm.YlGnBu(.9)
    
    #Bar plot
    fig, ax = plt.subplots()
    ax.grid(axis='x',alpha=0.3,zorder=0)

    ax.set_yticks(x)
    ax.set_yticklabels(['Clovelly', 'Coogee', 'Maroubra'])
    ax.barh(x, son,  color=a, label = 'Spring',zorder=3)
    ax.barh(x, djf, left=son,color=b,label='Summer',zorder=3) 
    ax.barh(x, mam, left=bars_1, color=c,label='Autumn',zorder=3)
    ax.barh(x, jja, left=bars_2, color=d,label = 'Winter',zorder=3)
    fig.legend(loc = 'upper center',ncol=4)
    ax.set_xlabel('Total occurences of beachings',fontsize = 10)

    #Set location of text
    y_loc = [0.98,1.98,2.98]    
    x_loc_son = [0.5*son[i] for i in range(0,3)]
    x_loc_djf = [son[i] + 0.5*djf[i] for i in range(0,3)]
    x_loc_mam = [son[i] + djf[i] + 0.4*mam[i] for i in range(0,3)]
    x_loc_jja = [son[i] + djf[i] + mam[i] + 0.4*jja[i] for i in range(0,3)]
    
    #hand written so that everyone rounds to 100%
    per_djf[0]=0.73
    
    #Add % in text
    for i in range(0,3):
        ax.text(x_loc_son[i]-1,y_loc[i],str(int(Decimal(per_son[i]*100).to_integral_value(rounding=ROUND_HALF_UP)))+'%',
                color = 'white',fontsize=10,rotation = 'vertical',verticalalignment='center',zorder=4)
    
        ax.text(x_loc_djf[i]-1,y_loc[i],str(int(Decimal(per_djf[i]*100).to_integral_value(rounding=ROUND_HALF_UP)))+'%',
                color = 'white',fontsize=10,rotation = 'vertical',verticalalignment='center',zorder=4)
        
        ax.text(x_loc_mam[i]-1,y_loc[i],str(int(Decimal(per_mam[i]*100).to_integral_value(rounding=ROUND_HALF_UP)))+'%',
                color = 'white',fontsize=10,rotation = 'vertical',verticalalignment='center',zorder=4)
    for i in range(1,3):
        
        ax.text(x_loc_jja[i]-1,y_loc[i],str(int(Decimal(per_jja[i]*100).to_integral_value(rounding=ROUND_HALF_UP)))+'%',
                color = 'white',fontsize=10,rotation = 'vertical',verticalalignment='center',zorder=4)
        
    fig.tight_layout(pad=2)
    
    fig.savefig(path_fig+'observed_season_bars.pdf')








def seasonal_cycle_plot(nb_beach,path_fig,
                        date_none,date_observed,
                        date,bluebottles,water_temp,
                        date_BOM,df_BOM,
                        ws, wd_meteo_deg):
    
    """
    Inputs : 
    - nb_beach (0 Clov 1 Coog 2 Mar), path_fig (folder to store figs)
    - date_none,date_observed from fct dates_observed_none()
    - date,bluebottles,water_temp from lifeguard obs
    - date_BOM,df_BOM daily BOM data grom get_kurnell_data_as_modified()
    - ws, wd_meteo_deg from BOM data
    
    
    Outputs : 
        - Plot of the seasonal cycle (at a weekly timescale) of beaching obs 
        - Over which is plotted the seasonal cycle (at a weekly timescale)
        of SST, cross shelf wind velocity, wind speed (and wind direction categories)
    
    """
    
    
    

    b=nb_beach
    location = ['Clovelly', 'Coogee', 'Maroubra']

    
    # ---- put in datafram and groupby year ----- #
    df_none = pd.DataFrame({'none' : date_none[b]})
    df_obs = pd.DataFrame({'obs' : date_observed[b]})
    
    none_yr_groups = df_none['none'].groupby(df_none['none'].dt.year).groups
    df_none_yr = np.array([df_none['none'][none_yr_groups[yr]] for yr in range(2016,2021)])
    
    obs_yr_groups = df_obs['obs'].groupby(df_obs['obs'].dt.year).groups
    df_obs_yr = np.array([df_obs['obs'][obs_yr_groups[yr]] for yr in range(2016,2021)])
    
    
    #----- compute seasonal cycle of beachings (weekly timescale) ------#
    none_all = []
    obs_all = []
    for d in range(0,len(df_obs_yr)):
        
        #group by week
        n = df_none_yr[d].groupby(df_none_yr[d].dt.week).count()
        o = df_obs_yr[d].groupby(df_obs_yr[d].dt.week).count()
        
        #when 0 observed for a month, the index disappears so we put it to 0    
        none = np.zeros((52)) #52 = nb of weeks in a year
        observed = np.zeros((52))
        
        normed_none = np.zeros((52))
        normed_observed = np.zeros((52))
        
        for i in range(1,53): 
        
            if np.isin(i, n.index):
                none[i-1] = n[i]
        
            if np.isin(i, o.index):
                observed[i-1] = o[i]
        
            normed_none[i-1] = none[i-1]/(none[i-1]+observed[i-1]) 
            normed_observed[i-1] = observed[i-1]/(none[i-1]+observed[i-1])
        
        none_all.append(normed_none)
        obs_all.append(normed_observed)
        
    none_all = np.array(none_all)
    obs_all = np.array(obs_all)
    
    
    #Take wind data and bb, sst data 
    date_wind = np.array([pd.to_datetime(d) for d in date_BOM])
    time = np.array([pd.to_datetime(d) for d in date[b][0:-1]]) #01/06/2018 --> 01/06/2019 685:1033
    bb = bluebottles[b][0:-1] #last index in excess
    dates_bb = date[b][0:-1]
    sst = water_temp[b][0:-1]
    
    #find index of BOM data of the period of interest
    i_time_BOM = np.isin(date_wind[:], time) 
    i_time_bb = np.isin(time[:], date_wind) 

    #Many to Observed
    bb2 = bb[i_time_bb]
    bb2[bb2==2] = 1
    
    
    #sst data in a dataframe
    wat_temp = sst[i_time_bb]
    w_tpr = pd.DataFrame({"sst": wat_temp})
    ww=w_tpr['sst']
    
    #take wind data  U and V for cross shelf calculation
    time_bb = dates_bb[i_time_bb]
    U = df_BOM['Wind_u_ms']
    V = df_BOM['Wind_v_ms']
    U = U[i_time_BOM] #U and V in oceao 
    V = V[i_time_BOM]
    
    rot_deg_angle = - 25
    Wind_u_rot = np.cos(rot_deg_angle * np.pi / 180) * U + np.sin(rot_deg_angle * np.pi / 180) * V;  #  cross-shelf 
    Wind_v_rot = - np.sin(rot_deg_angle * np.pi / 180) * U + np.cos(rot_deg_angle * np.pi / 180) * V;
    
    #to plot different coastline orientation in shaded
    rot_deg_angle = np.arange(-15,-30,-1)
    Wind_u_rot_2=[]
    for r in rot_deg_angle:
        Wind_u_rot_2.append(np.cos(r * np.pi / 180) * U + np.sin(r * np.pi / 180) * V);  #  cross-shelf 
    
    
    #take wind data for seasonality of wind speed and wind direction categories        
    wind_dir = wd_meteo_deg[i_time_BOM] 
    wind_speed = ws[i_time_BOM]
    time_wind = date_wind[i_time_BOM]

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    week=np.linspace(1,52,num=len(normed_observed))
    

    
    ##CROSS SHELF
    
    fig, (ax2,ax,ax_new) = plt.subplots(3,1,figsize=(7,10))
    ax1 = ax.twinx()
    ax.text(0,45,'B',fontweight='bold',fontsize=22)
    ax.set_ylabel('Frequency of beaching events [%]', fontsize=10)
    bins=np.arange(1,54)
    ax.set_xticks(bins[:-1])
    #ax.set_xlabel('Week #',fontsize=10)
    
    mean = np.nanmean(obs_all,axis=0)*100
    std = np.nanstd(obs_all,axis=0)*100
    ax.bar(week,mean,width=0.4,color='slategray',align='center',label='observed',alpha=0.8)

    #ax.plot(week,mean,color='black',label='observed',alpha=0.6)
    ax.fill_between(week, mean-std, mean+std,color='lightgrey',alpha=0.4)
    ax.set_ylim(0,np.nanmax(mean+std))
    
    plt.gca().yaxis.grid(True,alpha=0.5)
    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.plot(week,Wind_u_rot.groupby(Wind_u_rot.index.week).mean(),color='grey',zorder=1)
    
    for u in Wind_u_rot_2:
        ax1.plot(week,u.groupby(u.index.week).mean(),color='grey',zorder=1,alpha=0.1)
        ax1.scatter(week, u.groupby(u.index.week).mean(), 
                            color='lightgrey',
                           alpha=1,
                           zorder=2,
                           s=45)
        
    sc=ax1.scatter(week, Wind_u_rot.groupby(Wind_u_rot.index.week).mean(),
                   c=Wind_u_rot.groupby(Wind_u_rot.index.week).mean(), 
                   cmap='bwr_r', 
                   zorder=2,
                   s=45,
                   norm=MidpointNormalize(vcenter=0))
    
    axins = inset_axes(ax1,
                       width="50%",  # width = 5% of parent_bbox width
                       height="5%",  # height : 50%
                       loc='upper center',
                       bbox_to_anchor=(0.62, 0.57, 0.5, 0.5),
                       bbox_transform=ax1.transAxes,
                       borderpad=-0.1,
                       )
    
    cb = fig.colorbar(sc,cax=axins,orientation='horizontal',label='Cross shelf wind velocity [m.s$^{-1}$]')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax1.set_ylabel('Cross shelf wind velocity [m.s$^{-1}$]',fontsize=10)
    
    
    
    ## TEMPERATURE
    ax3 = ax2.twinx()
    #ax2.set_ylabel('Frequency of beaching events [%]', fontsize=10)
    bins=np.arange(1,54)
    ax2.text(0,45,'A',fontweight='bold',fontsize=22)

    ax2.set_xticks(bins[:-1])
    #ax2.set_xlabel('Week #',fontsize=10)
    ax2.bar(week,mean,width=0.4,color='slategray',align='center',label='observed',alpha=0.8)
    #ax2.plot(week,mean,color='black',label='observed',alpha=0.6)
    ax2.fill_between(week, mean-std, mean+std,color='lightgrey',alpha=0.4)
    ax2.set_ylim(0,np.nanmax(mean+std))

    
    plt.gca().yaxis.grid(True,alpha=0.5)
    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    
    ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax3.plot(week,ww.groupby(Wind_u_rot.index.week).mean(),color='grey',zorder=1)


    sc=ax3.scatter(week, ww.groupby(Wind_u_rot.index.week).mean(), 
                   c=ww.groupby(Wind_u_rot.index.week).mean(), 
                   cmap='gnuplot2_r', 
                   zorder=2,
                   s=45)
    
    axins = inset_axes(ax3,
                       width="50%",  # width = 5% of parent_bbox width
                       height="5%",  # height : 50%
                       loc='upper center',
                       bbox_to_anchor=(0.62, 0.57, 0.5, 0.5),
                       bbox_transform=ax3.transAxes,
                       borderpad=-0.1,
                       )
    
    cb = fig.colorbar(sc,cax=axins,orientation='horizontal',
                      label='Water temperature [°C]',
                      ticks = [17,18,19,20,21])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax3.set_ylabel('Water temperature [°C]',fontsize=10)
    
    
    ##WIND DIRECTION
    
    ax4 = ax_new.twinx()
    #ax_new.set_ylabel('Frequency of beaching events [%]', fontsize=10)
    bins=np.arange(1,54)
    ax_new.set_xticks(bins[:-1])
    ax_new.set_xlabel('Week #',fontsize=10)
    ax_new.text(0,45,'C',fontweight='bold',fontsize=22)

    ax_new.bar(week,mean,width=0.4,color='slategray',align='center',label='observed',alpha=0.8)
    #ax_new.plot(week,mean,color='black',label='observed',alpha=0.6)
    ax_new.fill_between(week, mean-std, mean+std,color='lightgrey',alpha=0.4)
    ax_new.set_ylim(0,np.nanmax(mean+std))

    
    plt.gca().yaxis.grid(True,alpha=0.5)
    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    
    ax_new.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax4.plot(week,wind_speed.groupby(Wind_u_rot.index.week).mean(),color='grey',zorder=1)
    
    u_mean = U.groupby(U.index.week).mean()
    v_mean = V.groupby(V.index.week).mean()
    (rho,phi) = cart2pol(u_mean,v_mean)
    direc_colors = rad2deg360(phi)
    
    clrs = np.chararray(direc_colors.shape,itemsize=20)
    clrs[np.logical_and(direc_colors<=101.25,direc_colors>11.25)]='mediumseagreen'
    clrs[np.logical_and(direc_colors<=191.25,direc_colors>101.25)]='lightskyblue'
    clrs[np.logical_and(direc_colors<=281.25,direc_colors>191.25)]='pink'
    clrs[np.logical_or(direc_colors<=11.25,direc_colors>281.25)]='orange'
    
    for i in range(1,len(clrs)):
        sc=ax4.scatter(week[i-1], wind_speed.groupby(Wind_u_rot.index.week).mean()[i], 
                       #c=wind_dir.groupby(Wind_u_rot.index.week).mean(), 
                       c = str(clrs[i])[2:-1], 
                       zorder=2,
                       s=45)


    ax4.set_ylabel('Wind speed [m.s$^{-1}$]',fontsize=10)
        
    fig.tight_layout(pad=4.0)
    fig.savefig(path_fig+'seasonal_cycle_'+str(location[nb_beach])+'.pdf')




def correlation_btwn_variables(nb_beach,path_fig,
                               date, bluebottles, water_temp,
                               date_BOM, df_BOM,
                               ws,wd_meteo_deg):
    
    """
    Inputs : 
    - nb_beach (0 Clov 1 Coog 2 Mar), path_fig (folder to store figs)
    - date,bluebottles,water_temp from lifeguard obs
    - date_BOM,df_BOM daily BOM data grom get_kurnell_data_as_modified()
    - ws, wd_meteo_deg from BOM data
    
    
    Outputs : 
        - correlation coefficients between bb and sst, cross shelf, along shelf winds, 
        wind_dir and wind_speed 
    
    """
    
    b=nb_beach
    location = ['Clovelly','Coogee','Maroubra']
    #Take data
    date_wind = np.array([pd.to_datetime(d) for d in date_BOM])
    time = np.array([pd.to_datetime(d) for d in date[b][0:-1]]) #01/06/2018 --> 01/06/2019 685:1033
    bb = bluebottles[b][0:-1] #last index in excess
    dates_bb = date[b][0:-1]
    sst = water_temp[b][0:-1]
    
    #find index of BOM data of the period of interest (both wind and lifeguard observations)
    i_time_BOM = np.isin(date_wind[:], time[:]) 
    i_time_bb = np.isin(time[:], date_wind) 

    #Many = Observed
    bb2 = bb[i_time_bb]
    bb2[bb2==2] = 1
    
    wat_temp = sst[i_time_bb]
    time_bb = dates_bb[i_time_bb]
        
    #Get wind data for ws and wd categories
    wind_dir = wd_meteo_deg[i_time_BOM] 
    wind_speed = ws[i_time_BOM]
    
    
    #Get wind data for cross shelf 
    U = df_BOM['Wind_u_ms']
    V = df_BOM['Wind_v_ms']
    
    U = U[i_time_BOM] #U and V in oceao 
    V = V[i_time_BOM]
    time_wind = date_wind[i_time_BOM]

    
    rot_deg_angle = - 25
    Wind_u_rot_25 = np.cos(rot_deg_angle * np.pi / 180) * U + np.sin(rot_deg_angle * np.pi / 180) * V;  #  cross-shelf 
    Wind_v_rot_25 = - np.sin(rot_deg_angle * np.pi / 180) * U + np.cos(rot_deg_angle * np.pi / 180) * V;
    
    
    df = pd.DataFrame({'bb':bb2,
                       
    'sst' : wat_temp,
    
    'cross shelf 25':Wind_u_rot_25,
    'along shelf 25' : Wind_v_rot_25,
    
    'wind_dir':wind_dir,
    'wind_speed':wind_speed})
    
    
    corr = df.corr()
    
    """svm = sns.heatmap(corr[['bb']].sort_values(by=['bb'],ascending=True), 
                      linewidths=0.1, square=True, 
                      cmap='jet', linecolor='white', annot=True)"""

    return corr['bb']

#------------------ SMALLER TIMESCALES PLOTS -----------------#
def chances_of_beachings_wd(wd, date_BOM, date_none, date_obs, beach):
    
    """
    
    Input: 
        - wd : wind direction array from BOM 
        - date_BOM : date array from BOM 
        - date_none : date array of no beaching events from dates_observed_none()
        - date_obs : date array of beaching events from dates_observed_none()
        - beach : choice of the beach (0 Clovelly 1 Coogee 2 Maroubra)

    Output: for each wind direction (NE,SE,SW,NW):
        % of chances to have beaching events the day after (observed)
        in the order 0 Clovelly 1 Coogee 2 Maroubra 
        if no Clovelly (e.g. in Summer) : 0 Coogee 1 Maroubra
    
    """
    dates_wind = np.array([pd.to_datetime(d) for d in date_BOM])
    
    NE=np.where(np.logical_and(wd>11.25, wd<=101.25))
    SE=np.where(np.logical_and(wd>101.25, wd<=191.25))
    SW=np.where(np.logical_and(wd>191.25, wd<=281.25))
    NW=np.where(np.logical_or(wd>281.25, wd<=11.25))
    
    date_obs_NE = date_obs[ np.isin(date_obs-datetime.timedelta(days=1), dates_wind[NE]) ]
    date_obs_SE = date_obs[ np.isin(date_obs-datetime.timedelta(days=1), dates_wind[SE]) ]
    date_obs_SW = date_obs[ np.isin(date_obs-datetime.timedelta(days=1), dates_wind[SW]) ]
    date_obs_NW = date_obs[ np.isin(date_obs-datetime.timedelta(days=1), dates_wind[NW]) ]

    date_none_NE = date_none[ np.isin(date_none-datetime.timedelta(days=1), dates_wind[NE]) ]
    date_none_SE = date_none[ np.isin(date_none-datetime.timedelta(days=1), dates_wind[SE]) ]
    date_none_SW = date_none[ np.isin(date_none-datetime.timedelta(days=1), dates_wind[SW]) ]
    date_none_NW = date_none[ np.isin(date_none-datetime.timedelta(days=1), dates_wind[NW]) ]
    
    chances_NE = len(date_obs_NE)/(len(date_obs_NE) + len(date_none_NE))
    chances_SE = len(date_obs_SE)/(len(date_obs_SE) + len(date_none_SE))
    chances_SW = len(date_obs_SW)/(len(date_obs_SW) + len(date_none_SW))
    chances_NW = len(date_obs_NW)/(len(date_obs_NW) + len(date_none_NW))

    return chances_NE, chances_SE, chances_SW, chances_NW


def summer_rose_plot_map(date_summer_stings,date_summer,bb_summer,date_BOM,ws,wd_meteo_deg,folder,path_data):
    """
    Input : 
    sting observations (date_summer_stings)
    lifeguard observations (dates and bb values) for summer
    date_BOM array for BOM data
    ws and wd from BOM data
    
    Output :  Map showing windroses of beaching conditions
    for all 3 beaches
        
    """
    
    #--- compute wind roses ----#
    wd1 = []
    ws1 =[]
    
    wd2 = []
    ws2 = []
    
    wd3 = []
    ws3 = []
    for nb_beach in range(0,3):
        location = ['Clovelly', 'Coogee', 'Maroubra']
        
        dates_bb = date_summer[nb_beach]
        bb = bb_summer[nb_beach]
        dates_stings = np.array([pd.to_datetime(d) for d in date_summer_stings[nb_beach]])
        dates_wind = np.array([pd.to_datetime(d) for d in date_BOM])
        
        one_day = datetime.timedelta(days=1)
        
        dates_beach = np.array([pd.to_datetime(d) for d in dates_bb])
        
        #We take the index of the day before a beaching 
        dates_observed = dates_beach[np.where(bb!=0)]-one_day
        dates_non = dates_beach[np.where(bb==0)]-one_day
        
        i_all = np.isin(dates_wind[:],dates_non)
        
        i_observed_plus1 = np.isin(dates_wind[:],dates_observed)
        
        i_stings = np.isin(dates_wind[:],dates_stings)
        
        wind_speed = ws[i_all]
        wind_speed_obs = ws[i_observed_plus1]
        ws_stings = ws[i_stings]
        
        direction = wd_meteo_deg[i_all]
        direction_obs = wd_meteo_deg[i_observed_plus1]
        wd_stings = wd_meteo_deg[i_stings]
        
        
        df1 = pd.DataFrame({"speed": wind_speed, "direction": direction})
        
        df2 = pd.DataFrame({"speed": wind_speed_obs, "direction": direction_obs})
        
        df3 = pd.DataFrame({"speed": ws_stings, "direction": wd_stings})

        wd2.append(df1['direction'])
        ws2.append(df1['speed'])
        
        wd1.append(df2['direction'])
        ws1.append(df2['speed'])
    
        wd3.append(df3['direction'])
        ws3.append(df3['speed'])                          
        
        
        
        
    #----- coastline ------
    
    
    coast_file = path_data+'eaccoast.dat'
    coast = np.loadtxt(coast_file)
    
    
    c_mar_0 = coast[:,0][np.logical_and(coast[:,1]>-33.9575,coast[:,1]<-33.9469)]
    c_mar_1 = coast[:,1][np.logical_and(coast[:,1]>-33.9575,coast[:,1]<-33.9469)]
    
    
    c_coog_0 = coast[:,0][np.logical_and(coast[:,1]>-33.9263,coast[:,1]<-33.9197)]
    c_coog_1 = coast[:,1][np.logical_and(coast[:,1]>-33.9263,coast[:,1]<-33.9197)]
    
    
    c_clov_0 = coast[:,0][np.logical_and(coast[:,1]>-33.9179,coast[:,1]<-33.9161)]
    c_clov_1 = coast[:,1][np.logical_and(coast[:,1]>-33.9179,coast[:,1]<-33.9161)]

        
    #--- plot --
    from mpl_toolkits.basemap import Basemap
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    
    c_beach = 'goldenrod'
    fig, ax = plt.subplots(figsize=(18,10))
    
    ax.plot(coast[:,0],coast[:,1],'k',lw=3)
    ax.set_xlim(151.24,151.37)
    ax.set_ylim(-33.9639,-33.90)
    ax.plot(c_mar_0[c_mar_0>151.26],c_mar_1[c_mar_0>151.26],
            c=c_beach,
            lw=3)
    
    ax.plot(c_coog_0,c_coog_1,
            c=c_beach,
            lw=3)
    
    ax.plot(c_clov_0[c_clov_0>151.275],c_clov_1[c_clov_0>151.275],
            c=c_beach,
            lw=3)
    
    ax.fill_betweenx(coast[:,1],coast[:,0],x2=151.2,
                     hatch='/',facecolor='None',
                     edgecolor='lightgrey',
                     alpha=0.9)
    
    
    axin = ax.inset_axes([0,0.8,0.15,0.2])
    axin.set_axis_off()
    
    m = Basemap(resolution='c',
                projection='merc',
                llcrnrlat = -50,
                llcrnrlon = 100,
                urcrnrlat = 0,
                urcrnrlon = 160) # lat 0, lon 0
    
    m.drawcoastlines(ax=axin,linewidth=2,zorder=0)
    lon = 151
    lat = -34
    x,y = m(lon, lat)
    axin.plot(x, y, linestyle='none',marker='o',markerfacecolor=c_beach, 
              markeredgecolor=c_beach,markersize=10)
    
    
    #maroubra
    bins = np.arange(0.01, 12, 2)
    rect=[0.6,0.15,0.21,0.21] 
    wa=WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.bar(wd2[2], ws2[2], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa.tick_params(labelleft=False, labelbottom=False)
    wa.legend(bbox_to_anchor=(1.2, 0.),title='Wind speed [m.s$^{-1}$]')
    wa.text(0.35,0.91,'NO BB',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa.transAxes,
            fontsize=14)
    
    rect1=[0.46, 0.15, 0.21, 0.21]
    wa1=WindroseAxes(fig, rect1)
    fig.add_axes(wa1)
    wa1.bar(wd1[2], ws1[2], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa1.tick_params(labelleft=False, labelbottom=False)
    wa1.text(0.35,0.91,'BB',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa1.transAxes,
            fontsize=14)

    rect3=[0.32, 0.15, 0.21, 0.21]
    wa3=WindroseAxes(fig, rect3)
    fig.add_axes(wa3)
    wa3.bar(wd3[2], ws3[2], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa3.tick_params(labelleft=False, labelbottom=False)
    wa3.text(0.35,0.91,'STINGS',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa3.transAxes,
            fontsize=14)    

    #coogee
    bins = np.arange(0.01, 12, 2)
    rect=[0.6,0.41,0.21,0.21] 
    wa=WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.bar(wd2[1], ws2[1], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa.tick_params(labelleft=False, labelbottom=False)
    wa.text(0.35,0.91,'NO BB',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa.transAxes,
            fontsize=14)
    
    rect1=[0.45, 0.41, 0.21, 0.21]
    wa1=WindroseAxes(fig, rect1)
    fig.add_axes(wa1)
    wa1.bar(wd1[1], ws1[1], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa1.tick_params(labelleft=False, labelbottom=False)
    wa1.text(0.35,0.91,'BB',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa1.transAxes,
            fontsize=14)
    
    rect3=[0.3, 0.41, 0.21, 0.21]
    wa3=WindroseAxes(fig, rect3)
    fig.add_axes(wa3)
    wa3.bar(wd3[1], ws3[1], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa3.tick_params(labelleft=False, labelbottom=False)
    wa3.text(0.35,0.91,'STINGS',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa3.transAxes,
            fontsize=14)    

    
    
    #clovelly
    bins = np.arange(0.01, 12, 2)
    rect=[0.65,0.655,0.21,0.21] 
    wa=WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.bar(wd2[0], ws2[0], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa.tick_params(labelleft=False, labelbottom=False)
    wa.text(0.35,0.91,'NO BB',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa.transAxes,
            fontsize=14)
    
    rect1=[0.51, 0.655, 0.21, 0.21]
    wa1=WindroseAxes(fig, rect1)
    fig.add_axes(wa1)
    wa1.bar(wd1[0], ws1[0], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa1.tick_params(labelleft=False, labelbottom=False)
    wa1.text(0.35,0.91,'BB',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa1.transAxes,
            fontsize=14)
    
    rect3=[0.37, 0.655, 0.21, 0.21]
    wa3=WindroseAxes(fig, rect3)
    fig.add_axes(wa3)
    wa3.bar(wd3[0], ws3[0], normed=True, opening=0.8, edgecolor='white',cmap=cmo.cm.ice,bins=bins)
    wa3.tick_params(labelleft=False, labelbottom=False)
    wa3.text(0.35,0.91,'STINGS',fontweight='bold',
            fontstyle='oblique',horizontalalignment='center',
            verticalalignment='center', transform=wa3.transAxes,
            fontsize=14)    

    
    
    
    
    x, y, arrow_length = 0.95, 0.97, 0.09
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=20,
                xycoords=ax.transAxes)
    
    #add text
    ax.text(151.268,-33.9145,'Clovelly',fontstyle='oblique',fontweight='bold',fontsize=15,c=c_beach)
    ax.text(151.257,-33.9224, 'Coogee',fontstyle='oblique',fontweight='bold',fontsize=15,c=c_beach)
    mr=ax.text(151.2525,-33.9525,'Maroubra',fontstyle='oblique',fontweight='bold',fontsize=15,c=c_beach)
    
    ax.annotate("",
                xy=(151.277,-33.9487), xycoords='data',
                xytext=(151.259,-33.9497), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3",
                                color='k',
                                linewidth=2),
                                zorder=1
                )
    
    ax.annotate("",
                xytext=(151.263,-33.9241), xycoords='data',
                xy=(151.274,-33.9289), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3",
                                color='k',
                                linewidth=2),
                                zorder=1
                )
                
    ax.annotate("",
                xytext=(151.274,-33.9122), xycoords='data',
                xy=(151.286,-33.9084), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle3",
                                color='k',
                                linewidth=2),
                                zorder=1
                )
#    fig.savefig(folder+'fig_map_rose.pdf')      
    
    
    





















