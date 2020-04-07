# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:21:12 2020

@author: hartmann
"""


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
        
    def explorer(self, sensorplots = False, sensorpdf = False):
        nan_vals = self.daten.isnull().sum()
        if sensorplots:
            k = 0
            for nfig in range (50):
                plt.figure(nfig)
                for nsub in range (5):
                    plt.subplot(5,1,nsub+1)
                    for nplot in range(3):
                        plt.plot(self.daten.iloc[:,k])
                    k += 1
        
        return nan_vals
        
    
    def handle_missing_values(self):
        '''method to deal with nan values'''
        # interpolates the nan values linearily with the subsequent values 
        # non interpolateable values are set to 0
        self.input = self.input.interpolate(method='linear', limit_direction='forward', axis=0)
        self.input = self.input.fillna(0)
        self.daten = pd.concat([self.input, self.label], axis = 1)
    
        
        
dataobj = Datencleaner(roh_daten)
nan_vals = dataobj.explorer(sensorplots = False)
dataobj.handle_missing_values()
data = dataobj.daten
test = dataobj.explorer(sensorplots = False)
if sum(test>0):
    print('''Not all missing values have been handled''')
else:
    data.to_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\Bachelorarbeit\\processed_data\\clean_data')
