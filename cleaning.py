# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:21:12 2020

@author: hartmann
"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


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

class Datencleaner:
    def __init__(self, daten):
        self.input = daten[list(daten.columns[1:243])]               
        self.label = daten['Column: 244 Locomotion']                     #label spalte 244
        self.daten = pd.concat([self.input, self.label], axis = 1) 
        

    def handle_missing_values(self):
        '''method to deal with nan values'''
        # interpolates the nan values linearily with the subsequent values 
        # non interpolateable values are set to 0
        self.input = self.input.interpolate(method='linear', limit_direction='forward', axis=0)
        self.input = self.input.fillna(0)
        self.daten = pd.concat([self.input, self.label], axis = 1)
    
    def nan_vals_check(self):
        nan_vals = self.daten.isna()
        check = np.where (nan_vals == True)
        if check [1] is None:
            return True
        else:
            return False
    
    def outliers(self):
        pass
        
    
        
        
dataobj = Datencleaner(roh_daten)
dataobj.handle_missing_values()
check = dataobj.nan_vals_check()
data = dataobj.daten
if check:
    data.to_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\clean_data')
else:
    print("not all NaN values have been imputed")
