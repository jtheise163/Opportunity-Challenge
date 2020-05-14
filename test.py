# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:49:38 2020

@author: hartmann
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

def generate(parents, sample):
    children = []
    if sample >= 0:
        for counter in range(sample):
            lin = np.random.rand(np.shape(parents[0]))
            nlin = np.ones(np.shape(parents[0]))# - lin
            child = np.dot(lin, parents[0]) + np.dot(nlin, parents[1])
            children.append(child)
        children = np.asarray(children)
        return nlin
    else:
        pass
        
    

train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')
#train_data = train_data[:,:,1:] #cutting of the timestamp
Data = []
counter = 0
ov_label = []
for window in train_data[:,:,:-1]:
    ov_label.append(democratic_Vote(train_data[counter,:,-1]))
    (label_list, label_count) = np.unique(np.int32(ov_label), return_counts = True)
    add = np.max(label_count) - label_count
    pifs = np.sum(window, axis = 0)
    Data.append(pifs)
    counter += 1
#    if counter%1000 == 0:
#        print(counter)

neighb = NearestNeighbors(n_neighbors=2).fit(Data)
distances, indices = neighb.kneighbors(Data)
counter = 0
samples_per_sample = []
for label in label_list:
    n_to_add = add[counter]
    samples_per_sample.append(int(n_to_add/label_count[counter]))
    
    counter += 1

add_train_data = []  
counter = 0   
for window in train_data:
    counter1 = 0
    for label in label_list:
        if democratic_Vote(window[:,-1]) == label:
            parents = [window, train_data[indices[counter,1],:,:]]
            add_train_data.append(generate(parents, samples_per_sample[counter1]))
add_train_data = np.asarray(add_train_data)