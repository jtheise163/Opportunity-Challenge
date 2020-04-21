# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:21:12 2020

@author: hartmann
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.signal



'''daten einlesen'''
name_path = 'C:\\Users\\hartmann\\Desktop\\Opportunity\\OpportunityUCIDataset\\dataset\\column_names.txt'
data_path = 'C:\\Users\\hartmann\\Desktop\\Opportunity\\OpportunityUCIDataset\\dataset/*.dat'
column_names = open(name_path)
column_names = column_names.read()
column_names = column_names.splitlines()
del column_names[0:2]                   # delete empty columns
del column_names[243:246]               # delete empty columns

data = [] # list to store data from path
i= 0

for filename in glob.glob(data_path):
    data.append(pd.read_csv(filename, header = 0, names = column_names, sep=' '))
    i += 1
#    if i == 1: # n_files read
#        break
    
roh_daten = pd.concat(data)
roh_daten = roh_daten.reset_index()  
del data

'''class to clean the data from missing values and outliers'''
class Datencleaner:
    def __init__(self, daten, column_names):
        self.input = daten[list(daten.columns[1:243])]               
        self.label = daten['Column: 244 Locomotion']                     #label spalte 244
        self.daten = pd.concat([self.input, self.label], axis = 1) 
        self.column_names = column_names
        
    def select_columns_opp(self):
        """Selection of the 113 columns employed in the OPPORTUNITY challenge
        :param data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """
    
        #                     included-excluded
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        self.daten = pd.DataFrame.drop(self.daten, self.column_names[features_delete + 1], axis = 'columns')

    def handle_missing_values(self, method = 'linear_interpolation', threshhold_missingness = 0.5 ):
        '''method to deal with nan values'''
        # features with missingness of data above 50% are now deleted
        n_rows, n_cols = np.shape(self.daten)
        for col in range(n_cols):
            n_nans = np.sum(self.daten.iloc[:,col].isna())
            percentage_nan = n_nans/n_rows
            if  percentage_nan > threshhold_missingness:
                self.daten.drop(self.columns_names[col])
        '''different methods to deal with missing data'''       
        if method == 'linear_interpolation':
            # interpolates the missing data in the sensorchannels linear over time with the subsequent values
            # if there are to much values missing subsequently then the missing values are filled with zeros
            self.input = self.input.interpolate(method='linear', limit_direction='forward', axis=0)
            self.input = self.input.fillna(0)
            self.daten = pd.concat([self.input, self.label], axis = 1)
        elif method == 'MEAN':
            # imputes the missing values with the mean of the feature
            mean = np.mean(self.daten)
            n_rows, n_cols = np.shape(self.daten)
            for col in range (n_cols):
                nan_pos = np.where(self.daten.iloc[:,col].isna())[0]
                if len(nan_pos) != 0:
                    self.daten.iloc[nan_pos,col] = mean[col]
        elif method == 'last_carried_forward':
            # imputes values by carrying the last value forward eg. transcribing the missing value with the last observed value
            n_rows, n_cols = np.shape(self.daten)
            for col in range(n_cols):
                nan_pos = np.where(self.daten.iloc[:,col].isna())
                for row in nan_pos:
                    self.daten.iloc[row,col] = self.daten.iloc[row -1 , col]
                       
            self.daten = self.daten.fillna(0)
            
            
           
    
    def nan_vals_check(self):
        '''checks dataframe for nan values'''
        nan_vals = self.daten.isna()
        check = np.where (nan_vals == True)
        if len(check[0]) == 0:
            return True
        else:
            return False
    
    def outliers(self):
#        self.daten, mean, std = self.z_transform(self.daten)
         pass
        
    def datafilter(self, mode='median', n_med = 3):
        '''filters the data'''
        if mode=='median':
            # mode median filters the data with a sliding median filter of stride 1 and a krenel size of :param: n_medx1
           self.daten.iloc[:,1:-2] = scipy.signal.medfilt(self.daten.iloc[:,1:-2], (n_med,1))
           
'''main script###########################################################################################################################################################'''
'''Hyperparameters'''
#filtersize
filtersize = 3   
#threshhold of missingnes for a feature to be dropped
threshhold_missingness = 0.5

'''data processing'''          
dataobj = Datencleaner(roh_daten, column_names)
dataobj.select_columns_opp()
a = dataobj.handle_missing_values(method = 'linear_interpolation', threshhold_missingness=0.5)
check = dataobj.nan_vals_check()
dataobj.datafilter(mode='median', n_med = filtersize)
data = dataobj.daten

'''checking and saving'''
# checking if all NaNs have been removed, if so file is saved in clean data file
if check:
    data.to_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\clean_data')
else:
    print("not all NaN values have been imputed")
