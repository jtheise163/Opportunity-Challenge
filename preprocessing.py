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

'''Hyperparameters sliding Window'''
window_size = 30
stride      = 15 # overlap of 50 %

'''data dir'''
data = pd.read_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\clean_data').iloc[:,1:]


'''Data Preprocessor class'''
class Dataprocesser:
    def __init__(self, data):
        self.data = data
        self.time = data.iloc[:, 0]
        self.time_diff = np.diff(self.time)
        self.freq = 1/np.mean(self.time_diff)
        
    def timegraph(self):
        '''shows graphically where timejumps are'''
        plt.plot(self.time)
        
    def find_timejumps(self):
        '''finds timejumps in the dataframe'''
        self.timejump = []
        i = 0
        for val in self.time_diff:
            if val <= 0:
                self.timejump.append(i)
            i += 1
        
    def split_at_timejumps(self):
        '''splits the dataframe in a list of dataframes that are continuous in time eg without time jumps'''
        self.find_timejumps()
        self.datalist = []
        self.datalist.append(self.data.iloc[0:self.timejump[0],:])
        for i in range(len(self.timejump)-1):
            split_of_data = self.data.iloc[self.timejump[i]:self.timejump[i+1],:]
            self.datalist.append(split_of_data)
    
    def sliding(self, dataframe, window_size, stride, shuffle = 'False'):
        '''puts a sliding window over a dataframe with
           :param: window size: kernel size of the sliding window
           :param: stride:      step size of the sliding window
           :boolean: shuffle:   shuffle the windows randomly for later machine learning algorithms'''
        n_windows = int((len(dataframe)-window_size +1 )/stride)
        windowed_data = np.zeros((n_windows, window_size, np.shape(dataframe)[1]))
        dataframe = dataframe.iloc[:int(len(dataframe)/window_size)*window_size + int(window_size/2),:] # cutting the end of the dataframe to achieve integer window number
        
        for i in range(n_windows):
             windowed_data[i,:,:] = dataframe.iloc[i*stride:i*stride+window_size,:]
             
        if shuffle == 'True':
                np.random.shuffle(windowed_data)
                
        return windowed_data


dataobj = Dataprocesser(data)
dataobj.split_at_timejumps()
datalist = dataobj.datalist
windowed_data_list = []
for data_piece in datalist:
    windowed_data = dataobj.sliding(data_piece, window_size, stride)
    for window in windowed_data:
        windowed_data_list.append(window)
    
    
windowed_data_list = np.asarray(windowed_data_list)
if shuffle:
    np.random.shuffle(windowed_data_list)
   

np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\windowed_data', windowed_data_list)    
    