# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:01:54 2020
#testdevelop
@author: hartmann
"""
import numpy as np

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

def PC_transform(X,U):
    '''computes linear coordinate system transformation
    :param:   X: 2-d Numpy array: datapoints
    :param:   U: 2-d Numpy array: Transformation-matrix
    :returns: pC: 2-d Numpy array: X in the coordinate system of U'''
    pC = np.dot(U.T, X.T)
    return pC.T
    

def PCA(data, explained_variance):
    '''computes linear PCA transformation
       :param:   data: 2-d Numpy array: datapoint x feature
       :returns: pC:   2-d Numpy array: datapoint x feature: data projected to the principal component axes
       :returns: U:    2-d Numpy array: feature x feature:   Principal components axes
       :returns: S:    1-d Numpy array: feature:             "Eigenvalues" of Principal component axes'''
    n_features = np.shape(data)[-1]
    data = np.reshape(data, (-1, n_features))
    data = z_score(data)
    Sigma = np.cov(data.T)
    U, S, V = np.linalg.svd(Sigma)
    pC = PC_transform(data, U)
    explained_variance_ratio = S/ len(S)
    explained_variance_sum = 0
    n_features_subset = 0
    while explained_variance_sum < explained_variance or n_features_subset == len(S):
        explained_variance_sum += explained_variance_ratio[n_features_subset] 
        n_features_subset += 1
    pC = pC[:,:n_features_subset]
    return pC, n_features_subset, U, S

def unwindow(windowed_data, stride):
    '''rearranges 3-d time window data into 2-d data
    :param:   windowed_data: 3-d Numpy array: windows x window_size x feature: data as time windows
    :param:   stride:        Integer-Scalar:  stepsizes of the time windows in the data
    :returns: data:          2-d Numpy array: datapoints x feature           : data'''
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
    '''puts a sliding window over an 2-d data array
    :param: data:        2-d Numpy array: datapoints x feature: data 
    :param: window_size: Integer Scalar:  size of the sliding window
    :param: stride:      Integer Scalar:  step size of the sliding window
    :param: shuffle:     Boolean       :  shuffle the windows randomly for later machine learning algorithms?'''
    n_windows = int((len(data)-window_size +1 )/stride)
    windowed_data = np.zeros((n_windows, window_size, np.shape(data)[1]))
    data = data[:int(len(data)/window_size)*window_size + int(window_size/2),:] # cutting the end of the dataframe to achieve integer window number
        
    for i in range(n_windows):
        windowed_data[i,:,:] = data[i*stride:i*stride+window_size,:]
             
    if shuffle:
        np.random.shuffle(windowed_data)
                
    return windowed_data

def PCA_pipeline(train_data, test_data, stride):
    #preparation for PCA on train_data
    train_data_unwindowed = unwindow(train_data, stride)
    #PCA on train_data
    principal_Component, n_features_subset, principalAxes, S = PCA(train_data_unwindowed, explained_variance_max)
    train_data_pC = sliding(principal_Component, window_size = window_size, stride = stride, shuffle = False)
    #PCA on test data
    test_data_pC = []
    for window in test_data:
        window = PC_transform(window, principalAxes)[:,:n_features_subset]
        test_data_pC.append(window)
    test_data_pC = np.asarray(test_data_pC)
    return test_data_pC, train_data_pC
    
    

'''Hyperparameters'''
# PCA
do_PCA = False
explained_variance_max = 0.99
stride = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\Hyperparameters\\stride.npy')
window_size = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\Hyperparameters\\window_size.npy')
# Train test split method
train_test_split = 's_split'     #'s_split', 'k_fold'

# simple train val test split
if train_test_split == 's_split':
    '''importing the data'''
    train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')
    train_data = train_data[:,:,1:] #cutting of the timestamp
    test_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\test_data.npy')
    test_data = test_data[:,:,1:]  #cutting of the timestamp
    if do_PCA:
        test_data, train_data = PCA_pipeline(test_data, train_data, stride)
        
    
    














# import the preprocessed data
# kfold-crossvalidation
elif train_test_split == 'k_fold':
    fold_set = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\fold_set.npy')
    
