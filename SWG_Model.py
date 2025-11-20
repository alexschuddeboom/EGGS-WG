#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint:disable=invalid-name
import os
import gc
import cftime
import xarray as xr
import numpy as np
from scipy.special import erf
import scipy.stats as scistat
import scipy

def Data_load(variable_name,Directory):
    '''Load in the simulation coefficeints from generated files'''
    start_year='1950'
    end_year='2022'
    year_str=start_year+'_'+end_year
    filename=Directory+variable_name+'_PreSim_Coeff_'+year_str+'.nc'
    tmp_data=xr.open_dataset(filename)
    tmp_data.drop_vars('Residual_Grid')
    return tmp_data

def XY_load(variable_list,Directory):
    '''Load in the simulation coupling coefficeints'''
    test_list=['t2m','d2m']
    hourly_list=[]
    for var_name in test_list:
        if var_name in variable_list:
            hourly_list.append(var_name)
    full_var_str=''
    h_list=range(0,len(hourly_list))
    for i in h_list:
        full_var_str=full_var_str+hourly_list[i]+'_'
    hourname=Directory+full_var_str+'XY_Matrix.nc'
    Hour_XY=xr.open_dataset(hourname)
    return Hour_XY

def Simulate_Day_Occur(V_1,prob_vector,Omega):
    '''Simulate Daily Occurrence'''
    random_vec=np.random.default_rng().multivariate_normal(
        mean=list(np.zeros(len(V_1))),cov=Omega,method='cholesky')
    V_2=random_vec<=prob_vector
    return V_2, random_vec

def Simulate_Day_Amount(Occurence_vector,Omega,Rng_vector,Coeffs,P_Vec):
    '''Simulate Daily precip amounts'''
        #Coeffs: [Alpha, Beta_1, Beta_2]
    random_vec=np.random.default_rng().multivariate_normal(
        mean=list(np.zeros(len(Occurence_vector))),cov=Omega,method='cholesky')
    TV3=Precipitation_Calculate(Coeffs[0],Coeffs[1],Coeffs[2],
                                Rng_vector,random_vec,P_Vec)
    return TV3*Occurence_vector

def Precipitation_Calculate(Alpha,Beta1,Beta2,occur_rng,amount_rng,P_Value):
    '''Mathematics to simulate precipitation value'''
    #Calculate the precipitation value
    U_prob=0.5*(1+erf(occur_rng/np.sqrt(2)))
    P_prob=0.5*(1+erf(P_Value/np.sqrt(2)))
    V_prob=0.5*(1+erf(amount_rng/np.sqrt(2)))
    Scale=Alpha*P_prob
    Beta=np.zeros(len(Scale))
    count=0
    for ind_val in Scale:
        if U_prob[count]>= ind_val:
            Beta[count]=Beta2[count]
        else:
            Beta[count]=Beta2[count] + 2*(Beta1[count]-Beta2[count])*(1-U_prob[count]/ind_val)
        count+=1
    return 1 - Beta*np.log(1 - V_prob)

def Simulation_Init(Input_params,Spin_Up_num):
    '''Initiate the simulation'''
    P_vector=np.zeros(Input_params['S_len'])
    Var_Vector=np.zeros(len(Input_params['H_XY']['X_matrix'].values[0,:,0]))
    I_P0=scistat.norm.ppf(Input_params['TP']['P0_Vector'].values[:,0],0,1)
    I_P1=scistat.norm.ppf(Input_params['TP']['P1_Vector'].values[:,0],0,1)
    HX=Input_params['H_XY']['X_matrix'].values[0,:,:]
    HY=Input_params['H_XY']['Y_matrix'].values[0,:,:]
    for i in range(0,Spin_Up_num):
        #First Initialize precipitation
        prob_vector=np.squeeze(I_P0*[abs(P_vector-1)] + I_P1*P_vector)
        [Update_vector,_]=Simulate_Day_Occur(P_vector,prob_vector,
                Input_params['TP']['Brissette_Occurrence_Omega'].values[:,:,0])
        P_vector=np.squeeze(Update_vector*1)
        #Next T 
        Day_var=np.zeros([len(Input_params['H_XY']['X_matrix'].values[0,:,0]),24])
        for i in range(0,24):
            var_random_vector=np.random.default_rng().standard_normal(len(Var_Vector))
            Var_Vector=np.matmul(HX,Var_Vector)+np.matmul(HY,var_random_vector)
            Day_var[:,i]=Var_Vector
    Init_vectors={'tp':P_vector,'var':Var_Vector}
    return Init_vectors


def Full_Simulate_Day(Sim_Params,Date,input_dict):
    '''Simulate a day of model data'''
    #Runs a day of simulation
    I_mon=Date.astype('datetime64[M]').astype(int) % 12
    P_vector=input_dict['tp']
    Var_Vector=input_dict['var']
    I_P0=scistat.norm.ppf(Sim_Params['TP']['P0_Vector'].values[:,I_mon],0,1)
    I_P1=scistat.norm.ppf(Sim_Params['TP']['P1_Vector'].values[:,I_mon],0,1)

    HX=Sim_Params['H_XY']['X_matrix'].values[I_mon,:,:]
    HY=Sim_Params['H_XY']['Y_matrix'].values[I_mon,:,:]
    #Simulate precipitation
    Params=[Sim_Params['TP']['Alpha_Vector'].values[:,I_mon],
            Sim_Params['TP']['Beta_1_Vector'].values[:,I_mon],
            Sim_Params['TP']['Beta_2_Vector'].values[:,I_mon]]
    prob_vector=np.squeeze(I_P0*[abs(P_vector-1)] + I_P1*P_vector)
    [Update_vector,RNG]=Simulate_Day_Occur(P_vector,prob_vector,
            Sim_Params['TP']['Brissette_Occurrence_Omega'].values[:,:,I_mon])
    P_vector=np.squeeze(Update_vector*1)
    Update_vector2=Simulate_Day_Amount(P_vector,
            Sim_Params['TP']['Brissette_Amounts_Omega'].values[:,:,I_mon],
            RNG,Params,prob_vector)
    amount_vector=np.squeeze(Update_vector2*1)
    #Next T 
    Day_var=np.zeros([len(Sim_Params['H_XY']['X_matrix'].values[0,:,0]),24])
    R_var=np.zeros([len(Sim_Params['H_XY']['X_matrix'].values[0,:,0]),24])
    varlen=len(Var_Vector)
    for i in range(0,24):
        var_random_vector=np.random.default_rng().standard_normal(varlen)
        Var_Vector=np.matmul(HX,Var_Vector)+np.matmul(HY,var_random_vector)
        Day_var[:,i]=Var_Vector
        R_var[:,i]=var_random_vector
    Res_vectors={'tp':P_vector,'var':Var_Vector}
    #Need to convert the residual values to the output variables
    t2m=np.zeros([len(P_vector),24])
    d2m=np.zeros([len(P_vector),24])
    #determine hour of year
    i_day=Date.astype('datetime64[D]')-Date.astype('datetime64[Y]')
    i_day=i_day.astype(int)
    if i_day==365:
        i_day=364
    t2m_mean_D=Sim_Params['Var_data']['t2m']['Fourier_Mean_Dry'].values[
        i_day*24:(i_day+1)*24,:]
    t2m_std_D=Sim_Params['Var_data']['t2m']['Fourier_Std_Dry'].values[
        i_day*24:(i_day+1)*24,:]
    d2m_mean_D=Sim_Params['Var_data']['d2m']['Fourier_Mean_Dry'].values[
        i_day*24:(i_day+1)*24,:]
    d2m_std_D=Sim_Params['Var_data']['d2m']['Fourier_Std_Dry'].values[
        i_day*24:(i_day+1)*24,:]
    t2m_mean_W=Sim_Params['Var_data']['t2m']['Fourier_Mean_Wet'].values[
        i_day*24:(i_day+1)*24,:]
    t2m_std_W=Sim_Params['Var_data']['t2m']['Fourier_Std_Wet'].values[
        i_day*24:(i_day+1)*24,:]
    d2m_mean_W=Sim_Params['Var_data']['d2m']['Fourier_Mean_Wet'].values[
        i_day*24:(i_day+1)*24,:]
    d2m_std_W=Sim_Params['Var_data']['d2m']['Fourier_Std_Wet'].values[
        i_day*24:(i_day+1)*24,:]

    for ihour in range(0,24):
        T_M_V=P_vector*t2m_mean_W[ihour,:]+np.abs(1-P_vector)*t2m_mean_D[ihour,:]
        T_S_V=P_vector*t2m_std_W[ihour,:]+np.abs(1-P_vector)*t2m_std_D[ihour,:]
        D_M_V=P_vector*d2m_mean_W[ihour,:]+np.abs(1-P_vector)*d2m_mean_D[ihour,:]
        D_S_V=P_vector*d2m_std_W[ihour,:]+np.abs(1-P_vector)*d2m_std_D[ihour,:]
        t2m[:,ihour]=Var_Vector[0:len(P_vector)]*T_S_V+T_M_V
        d2m[:,ihour]=Var_Vector[len(P_vector):len(P_vector)*2]*D_S_V+D_M_V
    Output_variables={'tp_o':P_vector,'tp_a':amount_vector,'t2m':t2m,
                      'd2m':d2m}
    return Output_variables,Res_vectors


def Alt_CDF_Calculate(in_res,tp_o,Prob_vec,W_Table,D_Table):
    '''Calculate variable associated with ECDF'''
    #Calculates the value given CDF variables
    prob=0.5*(1+erf(in_res/np.sqrt(2)))
    Ac=np.zeros(len(prob))
    range_iter=range(0,len(prob))
    for ind in range_iter:
        if tp_o[ind]:
            cdf=W_Table[ind,:]
        else:
            cdf=D_Table[ind,:]
        index=prob[ind]*1000
        dist_weight=(prob[ind]-Prob_vec[int(index)])/(0.001)
        Ac[ind]=cdf[int(index)+1]*dist_weight+cdf[int(index)]*(1-dist_weight)
    return Ac


def RH_Calculate(dtas,tas):
    '''calculate RH from d2m and t2m'''
    a1=610.78
    a3=17.27
    a4=35.86
    T0=273.16
    RH=[]
    Ev=a1*np.exp(a3*(dtas-T0)/(dtas-a4))
    Esv=a1*np.exp(a3*(tas-T0)/(tas-a4))
    RH=Ev/Esv
    return RH


def D2M_2_RH(in_data):
    '''Uses the t2m and d2m input data to calculate the Relative Humidity'''
    t2m=in_data['t2m']
    d2m=in_data['d2m']
    RH=RH_Calculate(d2m, t2m)
    ind=RH>=1
    RH[ind]=1
    RH=RH*100
    return RH

def Save_decade(Data,Filename):
    '''Save a decade of data to disk'''
    #Runs a day of simulation
    new_dates=[]
    for i in range(0,len(Data['Date'])):
        current_date=Data['Date'][i].astype('datetime64[D]')
        iyear=current_date.astype('datetime64[Y]').astype(int) + 1970
        imon=current_date.astype('datetime64[M]').astype(int) % 12 + 1
        day=current_date-current_date.astype('datetime64[M]') +1
        iday=day / np.timedelta64(1, 'D')
        new_dates.append(cftime.datetime(iyear,imon,iday))
    ds = xr.Dataset(
        data_vars={'Sim_Precipitation':(["simulation_time","station"], Data['tp_a'])},
        coords={'simulation_time':new_dates,'station':list(range(0,len(Data['d2m'][:,0,0])))})
    ds['hour']=range(0,24)
    ds['latitude']=Data['Lat']
    ds['longitude']=Data['Lon']
    ds['Station_Mask']=(('latitude','longitude'),Data['Station_Mask'])
    ds['Altitude']=(('latitude','longitude'),Data['Altitude'])
    ds['simulation_t2m']=(('simulation_time','hour','station'),np.swapaxes(Data['t2m'],0,2))
    ds['simulation_RH']=(('simulation_time','hour','station'),np.swapaxes(Data['RH'],0,2))
    ds['simulation_d2m']=(('simulation_time','hour','station'),np.swapaxes(Data['d2m'],0,2))
    ds.convert_calendar('standard',dim='simulation_time',use_cftime=True)
    if os.path.isfile(Filename):
        os.remove(Filename)
    ds.to_netcdf(Filename)
    ds.close()
    gc.collect()
    return


def Model_run(Directory,Sim_num,Simulation_start_year,Simulation_end_year,
              variable_list):
    '''Central Function for running the program'''
    #Controls the model run
    Spin_up_days=1000
    #Handle data loading
    L_Directory=Directory+'Data/'
    S_Directory=Directory+'Simulation/'
    P_data=xr.open_dataset(L_Directory+'A_Omega_1950_2022.nc')
    S_len=len(P_data['Station'])
    in_data={}
    for varname in variable_list:
        in_data[varname]=Data_load(varname,L_Directory)
    #XY Loading
    H_XY=XY_load(variable_list,L_Directory)
    Model_Params={'TP':P_data,'Var_data':in_data,'Var_list':variable_list,
                  'H_XY':H_XY,'S_len':S_len}
    #Initialization
    Init_data=Simulation_Init(Model_Params,Spin_up_days)
    Sim_Start_day=np.datetime64(str(Simulation_start_year)+'-01-01')
    Sim_End_day=np.datetime64(str(Simulation_end_year)+'-01-01')
    Altitude_file=L_Directory+'Altitude.nc'
    Alt=xr.open_dataset(Altitude_file)
    #Run through each simulation
    print('Starting Simulation Iteration #'+str(Sim_num+1))
    sim_dir=S_Directory+'Simulation_'+str(Sim_num+1)+'/'
    if not os.path.isdir(sim_dir):
        os.mkdir(sim_dir)
    Current_date=Sim_Start_day.astype('datetime64[D]')
    #Generate list of days to be simulated
    Current_Vectors=Init_data
    start_flag=True
    var_exist_flag=False
    #Need Save each decade out seperately
    while Current_date!=Sim_End_day:
        year=Current_date.astype('datetime64[Y]').astype(int)
        day=Current_date-Current_date.astype('datetime64[Y]') +1
        if np.mod(year,10)==0 and day==1:
            if start_flag:
                start_flag=False
                start_str=str(Simulation_start_year)
                start_day=Current_date.astype('datetime64[D]').astype(int)
                end_day=(Current_date.astype('datetime64[Y]')+10).astype(
                    'datetime64[D]').astype(int)
                F_end_day=(Sim_End_day.astype('datetime64[Y]')).astype(
                    'datetime64[D]').astype(int)
                if F_end_day<end_day:
                    end_day=F_end_day
                day_len=end_day-start_day
            else:
                #Save data
                end_str=str(year+1970)
                filename=sim_dir+'Simulation_'+start_str+'_'+end_str+'.nc'
                Full_data['Lat']=Model_Params['TP']['latitude'].values
                Full_data['Lon']=Model_Params['TP']['longitude'].values
                Full_data['Station_Mask']=Model_Params['TP']['Station_Mask'].values
                Full_data['Altitude']=Alt['altitude'].values
                RH=D2M_2_RH(Full_data)
                Full_data['RH']=RH
                print('Saving Sim#'+str(Sim_num+1)+ ' years '+start_str+' to '+end_str)
                Save_decade(Full_data,filename)
                var_exist_flag=False
                start_str=end_str
                start_day=Current_date.astype('datetime64[D]').astype(int)
                end_day=(Current_date.astype('datetime64[Y]')+10).astype(
                    'datetime64[D]').astype(int)
                F_end_day=(Sim_End_day.astype('datetime64[Y]')).astype(
                    'datetime64[D]').astype(int)
                if F_end_day<end_day:
                    end_day=F_end_day
                day_len=end_day-start_day
        #Simulate each day
        Output_data,Current_Vectors=Full_Simulate_Day(Model_Params,Current_date,Current_Vectors)
        if not var_exist_flag:
            Full_data={}
            Full_data['tp_o']=np.zeros([day_len,S_len])
            Full_data['tp_a']=np.zeros([day_len,S_len])
            Full_data['t2m']=np.zeros([S_len,24,day_len])
            Full_data['d2m']=np.zeros([S_len,24,day_len])
            Full_data['Date']=np.zeros([day_len])
            var_exist_flag=True
            Full_data['tp_o'][0,:]=Output_data['tp_o']
            Full_data['tp_a'][0,:]=Output_data['tp_a']
            Full_data['t2m'][:,:,0]=Output_data['t2m']
            Full_data['d2m'][:,:,0]=Output_data['d2m']
            Full_data['Date'][0]=np.array(Current_date)
            count=0
        else:
            count+=1
            Full_data['tp_o'][count,:]=Output_data['tp_o']
            Full_data['tp_a'][count,:]=Output_data['tp_a']
            Full_data['t2m'][:,:,count]=Output_data['t2m']
            Full_data['d2m'][:,:,count]=Output_data['d2m']
            Full_data['Date'][count]=np.array(Current_date)
        Current_date=Current_date+np.timedelta64(1, 'D')
    if var_exist_flag:
        end_str=Current_date.astype('datetime64[Y]').astype(str)
        filename=sim_dir+'Simulation_'+start_str+'_'+end_str+'.nc'
        Full_data['Lat']=Model_Params['TP']['latitude'].values
        Full_data['Lon']=Model_Params['TP']['longitude'].values
        Full_data['Station_Mask']=Model_Params['TP']['Station_Mask'].values
        Full_data['Altitude']=Alt['altitude'].values
        RH=D2M_2_RH(Full_data)
        Full_data['RH']=RH
        Save_decade(Full_data,filename)
    return

def Sim_Loader(Sim_Dir):
    '''Loads in an existing simulation output'''
    files=os.listdir(Sim_Dir)
    data_dict={}
    tmp_list=[]
    for filename in files:
        name=Sim_Dir+filename
        if 'tmp' not in name:
            tmp_list.append(filename)
            with xr.open_dataset(name,use_cftime=True,cache=True) as tmp_data:
                data_dict[filename]=tmp_data
                tmp_data.close()
    return tmp_list, data_dict


def Precipitation_Adjust(Base_S_dir,observation_dir,Sim_num):
    '''Adjust the precipitation amount simulation distribution'''
    from scipy.interpolate import CubicSpline
    O_filename=observation_dir+'daily_tp_1950_2022.nc'
    Obs_tp=xr.open_dataset(O_filename)
    t_obs_full_data=Obs_tp['tp'].values
    obs_full_data=t_obs_full_data[Obs_tp['Precipitation_mask'].values]
    #Run through each simulation
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        tmp_data=S_Data[S_Filenames[0]]
        tp=tmp_data['Sim_Precipitation'].values
        time=tmp_data['simulation_time'].values
        Sim_data={}
        Sim_data['Lat']=tmp_data['latitude'].values
        Sim_data['Lon']=tmp_data['longitude'].values
        if len(S_Filenames)>1:
            for z in range(1,len(S_Filenames)):
                tmp_data=S_Data[S_Filenames[z]]
                tmp_tp=tmp_data['Sim_Precipitation'].values
                tmp_time=tmp_data['simulation_time'].values
                tp=np.concatenate([tp,tmp_tp],axis=0)
                time=np.concatenate([time,tmp_time],axis=0)
        sim_full_data=tp[tp>1]
        #Calculate adjustment factors
        ipercent=list(range(0,101,5))
        obs_percent=np.percentile(obs_full_data,ipercent)
        sim_percent=np.percentile(sim_full_data,ipercent)
        diff_percent=sim_percent-obs_percent
        diff_spline=CubicSpline(sim_percent, diff_percent)
        #Now need to adjust each of the grid cells
        for j in range(0,len(S_Filenames)):
            tmp_data=S_Data[S_Filenames[j]].copy(deep='True')
            tmp_tp=tmp_data['Sim_Precipitation'].values
            for s_ind in range(0,len(tmp_tp[0,:])):
                nonzero_ind=np.logical_not(tmp_tp[:,s_ind]==0)
                gt_50=np.logical_not(np.logical_not(np.greater(tmp_tp[:,s_ind],50)))
                alt_list=[]
                for alt in range(0,int(np.sum(gt_50))):
                    alt_list.append(tmp_tp[int(np.where(gt_50)[0][alt]),s_ind]-diff_spline(50))
                tmp_tp[nonzero_ind,s_ind]=tmp_tp[nonzero_ind,s_ind]-diff_spline(tmp_tp[nonzero_ind,s_ind])
                tmp_tp[gt_50,s_ind]=alt_list
            S_Data[S_Filenames[j]].close()
            f_t_name=Sim_Dir+'tmp_'+S_Filenames[j]
            if os.path.isfile(f_t_name):
                os.remove(f_t_name)
            tmp_data.to_netcdf(f_t_name)
            tmp_data.close()
    return


def Temperature_Cycle_Adjust(Base_S_dir,observation_dir,Sim_num):
    '''Adjusts the diurnal cycle of temperature to match observations'''
    O_filename=observation_dir+'Detrended_t2m_1950_2022.nc'
    Obs_t2m=xr.open_dataset(O_filename,decode_timedelta=False)
    Obs_full_data=Obs_t2m['t2m'].values
    Obs_time=Obs_t2m['time'].values
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        tmp_data=S_Data[S_Filenames[0]]
        t2m=tmp_data['simulation_t2m'].values
        time=tmp_data['simulation_time'].values
        Sim_data={}
        Sim_data['Lat']=tmp_data['latitude'].values
        Sim_data['Lon']=tmp_data['longitude'].values
        Station_Mask=tmp_data['Station_Mask'].values
        if len(S_Filenames)>1:
            for z in range(1,len(S_Filenames)):
                tmp_data=S_Data[S_Filenames[z]]
                tmp_t2m=tmp_data['simulation_t2m'].values
                tmp_time=tmp_data['simulation_time'].values
                t2m=np.concatenate([t2m,tmp_t2m],axis=0)
                time=np.concatenate([time,tmp_time],axis=0)
        Sim_season_di=np.zeros([12,24])
        sim_mon=time.astype('datetime64[M]').astype(int) % 12
        Obs_season_di=np.zeros([12,24])
        Obs_mon=Obs_time.astype('datetime64[M]').astype(int) % 12
        lat_len=len(Obs_full_data[0,0,:,0])
        lon_len=len(Obs_full_data[0,0,0,:])
        for i in range(0,12):
            Obs_mon_mean=np.nanmean(Obs_full_data[Obs_mon==i,:,:,:])
            Sim_mon_mean=np.nanmean(t2m[sim_mon==i,:,:])
            for z in range(0,24):
                Sim_mon_data=t2m[sim_mon==i,z,:]
                Obs_mon_data=Obs_full_data[Obs_mon==i,z,:,:].astype('float64')
                nan_filter=np.isnan(Station_Mask[:,:])
                for j in range(0,lat_len):
                    for k in range(0,lon_len):
                        if nan_filter[j,k]:
                            Obs_mon_data[:,j,k]=np.nan
                Sim_season_di[i,z]=np.nanmean(Sim_mon_data)-Sim_mon_mean
                Obs_season_di[i,z]=np.nanmean(Obs_mon_data)-Obs_mon_mean
        Adj_vectors=Obs_season_di-Sim_season_di
        #Now need to adjust each of the grid cells
        for j in range(0,len(S_Filenames)):
            tmp_data=S_Data[S_Filenames[j]].copy(deep='True')
            tmp_t2m=tmp_data['simulation_t2m'].values
            tmp_time=tmp_data['simulation_time'].values   
            for time_ind in range(0,len(tmp_time)):
                tmp_mon=tmp_time[time_ind].month-1
                for i_hour in range(0,24):
                    tmp_t2m[time_ind,i_hour,:]+=Adj_vectors[tmp_mon,i_hour]
            S_Data[S_Filenames[j]].close()
            f_t_name=Sim_Dir+'tmp_'+S_Filenames[j]
            if os.path.isfile(f_t_name):
                os.remove(f_t_name)
            tmp_data.to_netcdf(f_t_name)
            tmp_data.close()
    return

def Sim_replace(Base_S_dir,Sim_num):
    '''Overwrite the original simulation files with
    the temporary simulation files'''
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        list_dir=os.listdir(Sim_Dir)
        for iname in list_dir:
            if 'tmp' not in iname:
                os.remove(Sim_Dir+iname)
                tmp_f_name=Sim_Dir+'tmp_'+iname
                ds=xr.open_dataset(tmp_f_name)
                ds.to_netcdf(Sim_Dir+iname)
                ds.close()
                os.remove(tmp_f_name)
    return


def metadata_handling(Base_F_dir,Sim_number):
    '''Handles all the metadata for the file saving to disk'''
    adjust_metadata(Base_F_dir, Sim_number)
    Sim_replace(Base_F_dir, Sim_number)
    add_alt_metadata(Base_F_dir, Sim_number)
    Sim_replace(Base_F_dir, Sim_number)
    return


def adjust_metadata(Base_S_dir,Sim_num):
    '''Transforms the time vector to joint time/day'''
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        for z in range(0,len(S_Filenames)):
            data=S_Data[S_Filenames[z]]
            #extract data
            tmp_time=data['simulation_time'].values
            tmp_tp=data['Sim_Precipitation'].values
            tmp_alt=data['Altitude'].values
            tmp_t2m=data['simulation_t2m'].values
            tmp_RH=data['simulation_RH'].values
            tmp_d2m=data['simulation_d2m'].values
            #adjust dates to vector
            joint_time=[]
            new_dates=[]
            for sim_day in tmp_time:
                for sim_hour in range(0,24):
                    tmp_time_val=np.datetime64(sim_day)+np.timedelta64(sim_hour,'h')
                    joint_time.append(tmp_time_val)
            for idat in range(0,len(joint_time)):
                current_date=joint_time[idat].astype('datetime64[h]')
                iyear=current_date.astype('datetime64[Y]').astype(int) + 1970
                imon=current_date.astype('datetime64[M]').astype(int) % 12 + 1
                day=current_date.astype('datetime64[D]')-current_date.astype('datetime64[M]') +1
                iday=day / np.timedelta64(1, 'D')
                hour=current_date-current_date.astype('datetime64[D]')
                ihour=hour.astype(int)
                new_dates.append(cftime.datetime(iyear,imon,iday,ihour))
            #reorder the data
            tmp_st=data['station'].values
            #Restructure the variables
            n_tmp_t2m=np.zeros([len(new_dates),len(tmp_st)])
            n_tmp_RH=np.zeros([len(new_dates),len(tmp_st)])
            n_tmp_d2m=np.zeros([len(new_dates),len(tmp_st)])
            for xval in range(0,len(tmp_time)):
                for yval in range(0,24):
                    n_tmp_t2m[xval*24+yval,:]=tmp_t2m[xval,yval,:]
                    n_tmp_RH[xval*24+yval,:]=tmp_RH[xval,yval,:]
                    n_tmp_d2m[xval*24+yval,:]=tmp_d2m[xval,yval,:]
            #Adjust station structure to lat lon structure
            Lat=data['latitude']
            Lon=data['longitude']
            Station_len=int(np.nanmax(data['Station_Mask'])+1)
            Station_Mask=data['Station_Mask']
            var_list=['tp_a','t2m','d2m','RH']
            var_output={}
            for var_name in var_list:
                if var_name=='tp_a':
                    tmp_grid_in=tmp_tp
                elif var_name=='t2m':
                    tmp_grid_in=n_tmp_t2m
                elif var_name=='d2m':
                    tmp_grid_in=n_tmp_d2m
                elif var_name=='RH':
                    tmp_grid_in=n_tmp_RH
                tmp_grid_out=np.ones([len(tmp_grid_in[:,0]),len(Lat),len(Lon)])*np.nan
                for ilat in range(0,len(Lat)):
                    for ilon in range(0,len(Lon)):
                        if not(np.isnan(Station_Mask[ilat,ilon])):
                            tmp_grid_out[:,ilat,ilon]=tmp_grid_in[:,int(Station_Mask[ilat,ilon].values)]
                var_output[var_name]=tmp_grid_out
            #Save the dataset to the disk
            ds = xr.Dataset(
                data_vars={'simulation_Precipitation':(["simulation_day","latitude",'longitude'], var_output['tp_a'])},
            coords={'simulation_day':tmp_time,'latitude':data['latitude'],'longitude':data['longitude']})
            ds['simulation_time']=new_dates
            ds['Altitude']=(('latitude','longitude'),tmp_alt)
            ds['simulation_t2m']=(('simulation_time','latitude','longitude'),var_output['t2m'])
            ds['simulation_RH']=(('simulation_time','latitude','longitude'),var_output['RH'])
            ds['simulation_d2m']=(('simulation_time','latitude','longitude'),var_output['d2m'])
            Filename=Sim_Dir+'tmp_'+S_Filenames[z]
            if os.path.isfile(Filename):
                os.remove(Filename)
            ds.to_netcdf(Filename)
            ds.close()
            gc.collect()
    return

def add_alt_metadata(Base_S_dir,Sim_num):
    '''Adds the metadata to simulation output'''
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        for z in range(0,len(S_Filenames)):
            data=S_Data[S_Filenames[z]]
            data['simulation_Precipitation']=data['simulation_Precipitation'].assign_attrs(
                units="mm",full_name='Simulated daily precipitation',
                description="9 am to 9 am simulation bounds following GHCN conventions")
            data['Altitude']=data['Altitude'].assign_attrs(
                units="m",full_name='Altitude Grid')
            data['simulation_t2m']=data['simulation_t2m'].assign_attrs(
                units="k",full_name='Simulated 2m air temperature',
                description="Can adjust to degrees Celsius by subtracting 273.15. This corresponds to an instantaneous measurement matching the time step.")
            data['simulation_d2m']=data['simulation_d2m'].assign_attrs(
                units="k",full_name='Simulated 2m dew point temperature',
                description="Can adjust to degrees Celsius by subtracting 273.15. This corresponds to an instantaneous measurement matching the time step.")
            data['simulation_RH']=data['simulation_RH'].assign_attrs(
                units="%",full_name='Simulated Relative Humidity',
                description="Estimated using t2m and d2m. This corresponds to an instantaneous measurement matching the time step.")
            data['latitude']=data['latitude'].assign_attrs(units="degrees")
            data['longitude']=data['longitude'].assign_attrs(units="degrees")
            data['simulation_time']=data['simulation_time'].assign_attrs(full_name='Simulated hour of the given day in UTC')
            if os.path.isfile(Sim_Dir+'tmp_'+S_Filenames[z]):
                os.remove(Sim_Dir+'tmp_'+S_Filenames[z])
            data.to_netcdf(Sim_Dir+'tmp_'+S_Filenames[z])
    return

def Nonstationairy_mode(Base_S_dir,observation_dir,Sim_num):
    '''Reintroduces the model forcing associated with time evolution'''
    def Line_based_correction(in_fit, in_data, in_d,dind_2020,maxd):
        '''make adjustments to the data based on the trend line'''
        in_d_index=in_d>maxd
        in_d[in_d_index]=max_d
        in_d_index=in_d<0
        in_d[in_d_index]=0
        line_vals = in_fit[0] + in_fit[1] * in_d
        ind_vals = in_fit[0] + in_fit[1] * dind_2020
        adjust_vals = ind_vals-line_vals
        out_data = in_data - adjust_vals
        return out_data
    #Load in observational trending data
    O_t2m_filename=observation_dir+'Detrended_t2m_1950_2022.nc'
    Obs_t2m=xr.open_dataset(O_t2m_filename,decode_timedelta=False)
    Obs_time=Obs_t2m['time'].values
    t2m_co=Obs_t2m['Detrend Coefficients'].values
    O_d2m_filename=observation_dir+'Detrended_d2m_1950_2022.nc'
    Obs_d2m=xr.open_dataset(O_d2m_filename,decode_timedelta=False)
    d2m_co=Obs_d2m['Detrend Coefficients'].values
    ind_2020=np.argwhere(Obs_time==np.datetime64('2020-01-01'))[0][0]
    #iterate through simulations
    for i in range(0,Sim_num):
        Sim_Dir=Base_S_dir+'Simulation/Simulation_'+str(i+1)+'/'
        S_Filenames,S_Data=Sim_Loader(Sim_Dir)
        for z in range(0,len(S_Filenames)):
            data=S_Data[S_Filenames[z]]
            #extract data
            tmp_time=data['simulation_time'].values
            tmp_t2m=data['simulation_t2m'].values
            tmp_d2m=data['simulation_d2m'].values
            d_values=tmp_time.astype('datetime64[D]').astype(int)-np.min(Obs_time.astype('datetime64[D]').astype(int))
            max_d=np.max(Obs_time.astype('datetime64[D]').astype(int))-np.min(Obs_time.astype('datetime64[D]').astype(int))
            new_t2m=np.ones(np.shape(tmp_t2m))*np.nan
            new_d2m=np.ones(np.shape(tmp_t2m))*np.nan
            for station in range(0,len(tmp_t2m[0,0,:])):
                for hour in range(0,24):
                    new_t2m[:,hour,station]=Line_based_correction(t2m_co, tmp_t2m[:,hour,station], d_values,ind_2020,max_d)
                    new_d2m[:,hour,station]=Line_based_correction(d2m_co, tmp_d2m[:,hour,station], d_values,ind_2020,max_d)
            tmp_data=data.copy(deep='True')
            tmp_data['simulation_t2m'].values=new_t2m
            tmp_data['simulation_d2m'].values=new_d2m
              
            f_t_name=Sim_Dir+'tmp_'+S_Filenames[z]            
            if os.path.isfile(f_t_name):
                os.remove(f_t_name)
            tmp_data.to_netcdf(f_t_name)
            tmp_data.close()
            data.close()
    return