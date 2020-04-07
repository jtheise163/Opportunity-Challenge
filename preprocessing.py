# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:34:57 2020

@author: hartmann
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''shuffle?'''
shuffle = True


data = pd.read_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\Bachelorarbeit\\processed_data\\clean_data').iloc[:,1:]



class Dataprocesser:
    def __init__(self, data):
        self.data = data
        self.time = data.iloc[:, 0]
        self.time_diff = np.diff(self.time)
        self.freq = 1/np.mean(self.time_diff)
        
    def timegraph(self):
        plt.plot(self.time)
        
    def find_timejumps(self):
        self.timejump = []
        i = 0
        for val in self.time_diff:
            if val <= 0:
                self.timejump.append(i)
            i += 1
        
    def split_at_timejumps(self):
        self.datalist = []
        self.datalist.append(self.data.iloc[0:timejump[0],:])
        for i in range(len(timejump)-1):
            split_of_data = self.data.iloc[timejump[i]:timejump[i+1],:]
            self.datalist.append(split_of_data)
    
    def sliding(self, dataframe, window_size, stride, shuffle = 'False'):
        
        n_windows = int((len(dataframe)-window_size +1 )/stride)
        windowed_data = np.zeros((n_windows, window_size, np.shape(dataframe)[1]))
        dataframe = dataframe.iloc[:n_windows*window_size,:]
        
        for i in range(n_windows):
             windowed_data[i,:,:] = dataframe.iloc[i*stride:i*stride+window_size,:]
             
        if shuffle == 'True':
                np.random.shuffle(windowed_data)
                
        return windowed_data

window_size = 30

dataobj = Dataprocesser(data)
dataobj.timegraph()
dataobj.find_timejumps()
timejump = dataobj.timejump
dataobj.split_at_timejumps()
datalist = dataobj.datalist
windowed_data_list = []
for data_piece in datalist:
    windowed_data = dataobj.sliding(data_piece, window_size, 15)
    for window in windowed_data:
        windowed_data_list.append(window)
    
    
windowed_data_list = np.asarray(windowed_data_list)
if shuffle:
    np.random.shuffle(windowed_data_list)
   

np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Bachelorarbeit\\processed_data\\windowed_data', windowed_data_list)    
    