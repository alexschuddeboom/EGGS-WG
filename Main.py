#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import DependencyCheck as DC
import shutil
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

ARGS = len(sys.argv) - 1
if ARGS == 0:
    print('No argument given for YAML file. Trying on disk version.')
    YAML_DIR = os.getcwd()+'/'
    YAML_FILE = 'Parameters.yaml'
    YAML_STR = YAML_DIR+YAML_FILE
    print(YAML_STR)
else:
    YAML_STR = sys.argv[1]
DEPENDENCY_FLAG = DC.DependencyCheck_Func()

if not DEPENDENCY_FLAG:
    sys.exit('Please install dependencies and re-run the program')
else:
    # Import the yaml data
    import yaml
    stream = open(YAML_STR, 'r', encoding='utf-8')
    Params = yaml.safe_load(stream)
    Params['Variables'] = ['tp', 't2m', 'd2m']
    # Check for lat/lon ordering
    lat = Params['Lat Bounds']
    lon = Params['Lon Bounds']
    if lat[1] > lat[0]:
        Params['Lat Bounds'] = [lat[1], lat[0]]
    if lon[1] > lon[0]:
        Params['Lon Bounds'] = [lon[1], lon[0]]
    # Check requested directory exists and create if missing
    Tmp_dir = Params['Data Output Directory']
    if not os.path.isdir(Tmp_dir):
        os.mkdir(Tmp_dir)
    Full_dir = Params['Data Output Directory']+Params['Simulation name']+'/'
    if not os.path.isdir(Full_dir):
        os.mkdir(Full_dir)
    Var_dir = Full_dir+'Data/'
    if not os.path.isdir(Var_dir):
        os.mkdir(Var_dir)
    Sim_dir = Full_dir+'Simulation/'
    if not os.path.isdir(Sim_dir):
        os.mkdir(Sim_dir)
    else:
        shutil.rmtree(Sim_dir)
        os.mkdir(Sim_dir)
    Extract_Directory = Var_dir+'netcdf/'
    if not os.path.isdir(Extract_Directory):
        os.mkdir(Extract_Directory)
    # Start the model building process
    import SWG_Build_Functions as SWG_B
    # Need to extract the initial variables from ERA5 data
    var_list = Params['Variables']
    if os.path.isdir(Extract_Directory):
        SWG_B.Data_Extract(Params['ERA5 Directory'], Extract_Directory,
                           Params['Lat Bounds'], Params['Lon Bounds'])
    for var_name in var_list:
        # Call individual variable runner for the var list if no local file or recalc
        filename = Var_dir+var_name+'_1950_2022.nc'
        if not(os.path.isfile(filename)) or Params['Regenerate Data']:
            print('Extracting '+var_name+' From ERA5 grib format')
            SWG_B.Variable_Extract(var_name, Extract_Directory,
                                   Var_dir, Params['Lat Bounds'], Params['Lon Bounds'])
        if var_name != 'tp':
            filename = Var_dir+'Detrended_'+var_name+'_1950_2022.nc'
            if not(os.path.isfile(filename)) or Params['Regenerate Data']:
                print('Detrending the  '+var_name+' variable')
                SWG_B.Variable_Detrender(var_name, Var_dir)

    ###########################################################################
    # First Build the Precipitation model components
    # Change tp from hourly to daily
    filename = Var_dir+'daily_tp_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Transforming hourly precipitation to daily')
        SWG_B.Daily_Precipitation_Extract(Var_dir)
    # Markov probabilities calculation
    filename = Var_dir+'Markov_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Calculating Markov Probabilities')
        SWG_B.Markov_Calculation(Var_dir)
    # Occurence Cross calculation
    filename = Var_dir+'Corr_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Calculating Precipitation Occurence Cross-Correlation Values')
        SWG_B.Cross_Correlator(Var_dir)
    # Occurence Omega calculation
    filename = Var_dir+'O_Omega_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Calculating Occurence Omega Matrix')
        SWG_B.O_Omega(Var_dir)
    # Amounts Coeff calculation
    filename = Var_dir+'tp_Params_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Precipitation Parameters Calculation')
        SWG_B.Amounts_Coeff(Var_dir)
    # Amounts Corr calculation
    filename = Var_dir+'Amounts_Corr_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Precipitation Amount Cross Correlation')
        SWG_B.Amounts_Corr(Var_dir)
    # Amounts Omega calculation
    filename = Var_dir+'A_Omega_1950_2022.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Calculating Amounts Omega Matrix')
        SWG_B.A_Omega(Var_dir)

    ###########################################################################
    # Var list calculate
    for var_name in var_list:
        if var_name in ('t2m', 'd2m'):
            # Now need to prep t2m and d2m
            filename = Var_dir+var_name+'_PreSim_Coeff_1950_2022.nc'
            if not(os.path.isfile(filename)) or Params['Regenerate Data']:
                print('Building '+var_name+' simulation parameters')
                SWG_B.Fourier_Generation(Var_dir, var_name)

    # X and Y matrix calculate
    test_list = ['t2m', 'd2m']
    hourly_list = []
    for var_name in test_list:
        if var_name in var_list:
            hourly_list.append(var_name)
    FULL_VAR_STR = ''
    RANGE_LIST = range(0, len(hourly_list))
    for i in RANGE_LIST:
        FULL_VAR_STR = FULL_VAR_STR+hourly_list[i]+'_'
    filename = Var_dir+FULL_VAR_STR+'XY_Matrix.nc'
    if not(os.path.isfile(filename)) or Params['Regenerate Data']:
        print('Generating XY hourly matrix')
        SWG_B.XY_Hourly(Var_dir, hourly_list)
    ###########################################################################
    # Orography prep
    SWG_B.orography_prep(
        Params['ERA5 Directory'], Var_dir, Params['Lat Bounds'], Params['Lon Bounds'])
    ###########################################################################
    # Run the model
    import SWG_Model as SWG_M
    SWG_M.Model_run(Full_dir,
                    Params['Number of Simulations'], Params['Simulation Start Year'],
                    Params['Simulation End Year'], test_list)
    if Params['Variable Adjust']:
        SWG_M.Precipitation_Adjust(Full_dir, Var_dir, Params['Number of Simulations'])
        SWG_M.Sim_replace(Full_dir, Params['Number of Simulations'])
        SWG_M.Temperature_Cycle_Adjust(Full_dir, Var_dir, Params['Number of Simulations'])
        SWG_M.Sim_replace(Full_dir, Params['Number of Simulations'])
    SWG_M.metadata_handling(Full_dir, Params['Number of Simulations'])
