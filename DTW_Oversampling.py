# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:02:09 2020

@author: hartmann
"""

import numpy as np
from fastdtw import fastdtw
import matplotlib.pyplot as plt

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

def z_score(data):
    '''computes z-scores of data 
       every feature has a mean of 0 and a standard deviation of 1
       :param:   data:   2-d Numpy array: datapoint x feature
       :returns: z-score:2-d Numpy array: datapoint x feature (mean = 0, std=1) '''
    mean = np.mean(data, axis = 0) #columnswise mean and standard_deviation
    std = np.std(data, axis = 0)
    data = (data - mean)/std
    data = np.nan_to_num(data)
    return data


def z_score_pipeline(train_data, test_data):
    train_data_z = [] 
    counter = 0
    for window in train_data[:,:,:-1]:
        window = z_score(window)
        window = np.concatenate((window, train_data[counter, :, -1].reshape((-1,1))), axis = 1)
        train_data_z.append(window)
        counter += 1
    train_data_z = np.asarray(train_data_z)
    
#    test_data_z = [] 
#    counter = 0
#    for window in test_data[:,:,:-1]:
#        window = z_score(window)
#        window = np.concatenate((window, test_data[counter, :, -1].reshape((-1,1))), axis = 1)
#        test_data_z.append(window)
#        counter += 1
#    test_data_z = np.asarray(test_data_z)
    return train_data_z#
train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')
train_data = z_score_pipeline(train_data,[])

n_samples = 10#np.shape(train_data)[0]
n_features = np.shape(train_data)[2]-1
distance_matrix = []

for i in range(n_samples):
    for j in range(n_samples):
        if (democratic_Vote(train_data[i,:,-1])==democratic_Vote(train_data[j,:,-1])) and (democratic_Vote(train_data[i,:,-1])==0):
            for feature in range(n_features):
                distance, path = fastdtw(train_data[i,:,feature], train_data[j,:,feature])
                distance_matrix.append(distance)
                
                
    
    