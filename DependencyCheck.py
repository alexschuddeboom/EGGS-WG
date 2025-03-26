#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:43:50 2023

@author: alex
"""

def DependencyCheck_Func():
    flag=True
    try:
        import numpy
    except ModuleNotFoundError:
        print('Numpy is not installed')
        flag=False
    try:
        import scipy
    except ModuleNotFoundError:
        print('scipy is not installed')
        flag=False
    try:
        import netCDF4
    except ModuleNotFoundError:
        print('netCDF4 is not installed')
        flag=False
    try:
        import cftime
    except ModuleNotFoundError:
        print('cftime is not installed')
        flag=False    
    try:
        import os
    except ModuleNotFoundError:
        print('OS is not installed')
        flag=False
    try:
        import xarray
    except ModuleNotFoundError:
        print('xarray is not installed')
        flag=False
    try:
        import calendar
    except ModuleNotFoundError:
        print('calender is not installed')
        flag=False
    try:
        import yaml
    except ModuleNotFoundError:
        print('pyyaml is not installed')
        flag=False  
    try:
        import cfgrib
    except ModuleNotFoundError:
        print('cfgrib is not installed. CFgrib can cause compatability issues on install, makes sure to install last.')
        flag=False   
    return flag