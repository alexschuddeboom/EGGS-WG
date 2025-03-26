# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:54:20 2023

@author: Alex
"""
import numpy as np
from calendar import monthrange
import cdsapi

c = cdsapi.Client()

year_start=1950
year_end=2022

year_list=[]
for year in range(year_start,year_end+1):
    year_list.append(str(year))

mon_list=[]
for i in range(1,13):
    if i>9:
        mon_list.append(str(i))
    else:
        mon_list.append('0'+str(i))
        
day_list=[]
for i in range(1,32):
    if i>9:
        day_list.append(str(i))
    else:
        day_list.append('0'+str(i))

for year in year_list:
    count=0
    leap_flag=np.mod(int(year),4)==0
    for mon in mon_list:
        
        mon_len=monthrange(int(year), count+1)[1]
        count+=1
        tmp_day_list=day_list[0:mon_len]
        
        tmp_str='ERA_Land_Multi_var_'+year+'_'+mon+'.grib'
	
        dataset='reanalysis-era5-land'
        request={'variable': ['2m_dewpoint_temperature','2m_temperature', 'total_precipitation'],
                'year': year,
                'month': mon,
                'day': tmp_day_list,
                'time': ['00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',],
                'area': [-33.4, 165.2, -47.5, 179.6,],
                'format': 'grib',
    		"download_format": "unarchived"
            }
        c.retrieve(dataset,request,tmp_str)