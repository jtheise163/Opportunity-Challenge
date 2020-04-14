# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:01:54 2020

@author: hartmann
"""
import numpy as np
import sklearn

def z_score(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    data = (data - mean)/std
    data = np.nan_to_num(data)
    return data

def PC_transform(X,U):
    pC = np.dot(U.T, X.T)
    return pC.T
    

def PCA(data):
    n_features = np.shape(data)[-1]
    data = np.reshape(data, (-1, n_features))
    data = z_score(data)
    Sigma = np.cov(data.T)
    U, S, V = np.linalg.svd(Sigma)
    pC = PC_transform(data, U)
    return pC, U, S, V

def unwindow(windowed_data, stride):
    n_windows, window_size, n_features = np.shape(windowed_data)
    data_except_first_window = []
    first_loop = True
    for window in windowed_data:
        if first_loop:
            data_first_window = window
            first_loop = False
        else:
            data_except_first_window.append(window[window_size-stride:,:])
    data_except_first_window = np.asarray(data_except_first_window)
    data_except_first_window = np.reshape(data_except_first_window, (-1, n_features))
    data_first_window = np.asarray(data_first_window)
    data = np.vstack((data_first_window, data_except_first_window))
    return data

def sliding(data, window_size, stride, shuffle = False):
        '''puts a sliding window over a data array with
          :param: window size: kernel size of the sliding window
          :param: stride:      step size of the sliding window
          :param: shuffle:     shuffle the windows randomly for later machine learning algorithms'''
        n_windows = int((len(data)-window_size +1 )/stride)
        windowed_data = np.zeros((n_windows, window_size, np.shape(data)[1]))
        data = data[:int(len(data)/window_size)*window_size + int(window_size/2),:] # cutting the end of the dataframe to achieve integer window number
        
        for i in range(n_windows):
             windowed_data[i,:,:] = data[i*stride:i*stride+window_size,:]
             
        if shuffle:
                np.random.shuffle(windowed_data)
                
        return windowed_data
    
    

'''Hyperparameters'''
# PCA
explained_variance_max = 0.99
stride = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\Hyperparameters\\stride.npy')
window_size = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\Hyperparameters\\window_size.npy')

train_test_split = 's_split'     #'s_split', 'k_fold'

#import the preprocessed data
# simple train val test split
if train_test_split == 's_split':
    train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')
    train_data = train_data[:,:,1:]
    test_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\test_data.npy')
    test_data = test_data[:,:,1:]
    train_data_unwindowed = unwindow(train_data, stride)
    pC, U, S, V = PCA(train_data_unwindowed)
    explained_variance_ratio = S/len(S)
    explained_variance_sum = 0
    n_features_subset = 0
    while explained_variance_sum < explained_variance_max or n_features_subset == len(S):
        explained_variance_sum += explained_variance_ratio[n_features_subset] 
        n_features_subset +=1
    pC = pC[:,:n_features_subset]
    train_data_pC = sliding(pC, window_size = window_size, stride = stride, shuffle = False)
    test_data_pC = []
    for window in test_data:
        window = PC_transform(window, U)[:,:n_features_subset]
        test_data_pC.append(window)
    test_data_pC = np.asarray(test_data_pC)
    
    














# import the preprocessed data
# kfold-crossvalidation
elif train_test_split == 'k_fold':
    fold_set = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\fold_set.npy')
    
