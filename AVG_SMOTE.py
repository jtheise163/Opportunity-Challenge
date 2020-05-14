# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:49:38 2020

@author: hartmann
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
#import scipy

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

def generate(parent1, parent2):
    n_timesteps, n_features = np.shape(parent1)
    random_array = np.random.rand(n_timesteps, n_features)
    inv_random_array = np.ones((n_timesteps, n_features)) - random_array
    child = np.multiply(random_array, parent1) + np.multiply(inv_random_array, parent2)
    return child

def matchmaker():
    pass



def AVG_SMOTE(train_data):
    label_list = [democratic_Vote(train_data[i,:,-1]) for i in range(np.shape(train_data)[0])]
    label, counts = np.unique(label_list, return_counts = True)
    n_samples = max(counts)-counts
    samples_per_sample = np.int16(np.round(n_samples/counts))
    
    point_in_feature_space = []
    for window in train_data[:,:,:-1]:
        point_in_feature_space.append(np.mean(window, axis = 0))
    
    neighb = NearestNeighbors(n_neighbors=2).fit(point_in_feature_space)
    distances, indices = neighb.kneighbors(point_in_feature_space)
    
    counter = 0
    synthetic_data = []
    syn_label = []
    for n_samples in samples_per_sample:
        if n_samples > 0:
            window_nr = 0
            for window in train_data:
                if democratic_Vote(window[:,-1]) == label[counter]:
                    for _ in range(n_samples):
                        child = generate(window[:,:-1], train_data[indices[window_nr,1],:,:-1])
                        syn_label.append(window[:,-1])
                        synthetic_data.append(child)
                window_nr += 1
    
    
        counter += 1
        print(counter)
    synthetic_data = np.asarray(synthetic_data)
    syn_label = np.asarray(syn_label)
    print(np.shape(syn_label))
    print(np.shape(synthetic_data))
    synthetic_data = np.concatenate((synthetic_data, np.reshape(syn_label,(-1,np.shape(train_data)[1],1))), axis = 2)
    synthetic_data = np.concatenate((synthetic_data, train_data), axis = 0)
    return synthetic_data

train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')
data = AVG_SMOTE(train_data)

















