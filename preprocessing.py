# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:34:57 2020

@author: hartmann
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''Hyperparameters sliding Window'''
window_size = 30
stride      = 15 # overlap of 50 %
np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Hyperparameters\\stride', stride) 
np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Hyperparameters\\window_size', window_size) 

'''data dir'''
data = pd.read_csv('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\clean_data').iloc[:,1:]


'''Data Preprocessor class'''
class Dataprocessor:
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
    
    @staticmethod
    def sliding(dataframe, window_size, stride, shuffle = False):
        '''puts a sliding window over a dataframe with
           :param: window size: kernel size of the sliding window
           :param: stride:      step size of the sliding window
           :boolean: shuffle:   shuffle the windows randomly for later machine learning algorithms'''
        n_windows = int((len(dataframe)-window_size +1 )/stride)
        windowed_data = np.zeros((n_windows, window_size, np.shape(dataframe)[1]))
        dataframe = dataframe.iloc[:int(len(dataframe)/window_size)*window_size + int(window_size/2),:] # cutting the end of the dataframe to achieve integer window number
        
        for i in range(n_windows):
             windowed_data[i,:,:] = dataframe.iloc[i*stride:i*stride+window_size,:]
             
        if shuffle:
                np.random.shuffle(windowed_data)
                
        return windowed_data
    
    @staticmethod
    def list_to_timewindow(datalist, shuffle = False):
        windowed_data_list = []
        for data_piece in datalist:
            windowed_data = Dataprocessor.sliding(data_piece, window_size, stride)
            for window in windowed_data:
                windowed_data_list.append(window)
        windowed_data_list = np.asarray(windowed_data_list)
        if shuffle:
            np.random.shuffle(windowed_data_list)
        return windowed_data_list
    

def train_test_split(data, percentage):
    n_windows = np.shape(data)[0]
    n_test = int(percentage * n_windows)
    random_index = np.random.randint(n_windows, size=n_test)
    train_data = np.delete(data, random_index, axis = 0)
    test_data = data[random_index, :, :]
    return train_data, test_data

def k_fold_x_val(data, k):
    np.random.shuffle(data)
    n_windows, window_size, n_features = np.shape(data)
    fold_size = int(n_windows/k)
    data = data[:fold_size*k,:,:]
    fold_set = np.zeros((k, fold_size, window_size, n_features))
    for fold in range(k):
        fold_set[fold,:,:,:] = data[fold*fold_size:(fold+1)*fold_size,:,:]
    return fold_set

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

def naive_balancer(windowed_data_list):
    vote = []
    for window in windowed_data_list:
        vote.append(democratic_Vote(window[:,-1]))
    (label_list, label_count) = np.unique(np.int32(vote), return_counts = True)
    add = np.max(label_count) - label_count
    label_counter = 0
    for label in label_list:
        label_index = np.where(vote == label)[0]
        random_index = np.random.choice(label_index, (add[label_counter]))
        windowed_data_list = np.concatenate((windowed_data_list, windowed_data_list[random_index,:,:]),axis=0)
        label_counter += 1
        
    return windowed_data_list


balanced = True
dataobj = Dataprocessor(data)
dataobj.split_at_timejumps()
datalist = dataobj.datalist
windowed_data_list = Dataprocessor.list_to_timewindow(datalist, shuffle = True)
if balanced:
    windowed_data_list = naive_balancer(windowed_data_list)

   
train_data, test_data = train_test_split(windowed_data_list, 0.2)
fold_set = k_fold_x_val(windowed_data_list, 10)
np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data', train_data)    
np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\test_data', test_data)    
np.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\fold_set', fold_set)    
























    