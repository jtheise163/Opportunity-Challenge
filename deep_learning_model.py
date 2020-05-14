# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:01:54 2020
#testdevelop
@author: hartmann
"""
import numpy as np
import keras
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors

def z_score(data):
    '''computes z-scores of data 
       every feature has a mean of 0 and a standard deviation of 1
       :param:   data:   2-d Numpy array: datapoint x feature
       :returns: z-score:2-d Numpy array: datapoint x feature (mean = 0, std=1) '''
    e = 1e-10
    mean = np.mean(data, axis = 0) #columnswise mean and standard_deviation
    std = np.std(data, axis = 0)
    data = (data - mean)/(std+e)
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
#    n_features = np.shape(data)[-1]
#    data = np.reshape(data, (-1, n_features))
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
    principal_Component, n_features_subset, principalAxes, S = PCA(train_data_unwindowed[:,:-1], explained_variance_max)
    train_data = np.hstack((principal_Component, train_data_unwindowed[:,-1].reshape((-1,1))))
    train_data_pC = sliding(train_data, window_size = window_size, stride = stride, shuffle = False)
    #PCA on test data
    test_data_pC = []
    counter = 0
    for window in test_data[:,:,:-1]:
        window = z_score(window)
        window = PC_transform(window, principalAxes)[:,:n_features_subset]
        window = np.concatenate((window, test_data[counter,:,-1].reshape(-1,1)), axis = 1)
        test_data_pC.append(window)
        counter += 1
    test_data_pC = np.asarray(test_data_pC)
    #print(np.shape(test_data_pC), np.shape(test_data[:,:,-1].reshape((-1, window_size, 1))))
    return train_data_pC, test_data_pC

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

#def generate(parent1, parent2):
#    n_timesteps, n_features = np.shape(parent1)
#    random_array = np.random.rand(n_timesteps, n_features)
#    inv_random_array = np.ones((n_timesteps, n_features)) - random_array
#    child = np.multiply(random_array, parent1) + np.multiply(inv_random_array, parent2)
#    return child

def generate(parent1, parent2):
    n_timesteps, n_features = np.shape(parent1)
    random = np.random.rand()
    inv_random = 1 - random
    child = random * parent1 + inv_random * parent2
    return child

   
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

#
#def ADASYN(data):
#    neighbours = 
#    pass

def z_score_pipeline(train_data, test_data):
    train_data_z = [] 
    counter = 0
    for window in train_data[:,:,:-1]:
        window = z_score(window)
        window = np.concatenate((window, train_data[counter, :, -1].reshape((-1,1))), axis = 1)
        train_data_z.append(window)
        counter += 1
    train_data_z = np.asarray(train_data_z)
    
    test_data_z = [] 
    counter = 0
    for window in test_data[:,:,:-1]:
        window = z_score(window)
        window = np.concatenate((window, test_data[counter, :, -1].reshape((-1,1))), axis = 1)
        test_data_z.append(window)
        counter += 1
    test_data_z = np.asarray(test_data_z)
    return train_data_z, test_data_z

def one_hot_builder(label_array, n_classes, window_size, mode = 'many_to_many'):
    if mode == 'many_to_one':
        label_array = stats.mode(label_array, axis =1)[0]
        label_array = label_array.reshape((-1))
        cat_label = []
        for label in label_array:
            if label == -1:
                cat_label.append([0,0,0,0])
            else:
                cat_label.append(tf.keras.utils.to_categorical(label,4))
        cat_label = np.asarray(cat_label)
        cat_label = cat_label.reshape((-1, n_classes))
    else:
        label_array = label_array.reshape((-1))
        cat_label = []
        for label in label_array:
            if label == -1:
                cat_label.append([0,0,0,0])
            else:
                cat_label.append(tf.keras.utils.to_categorical(label, 4))
        cat_label = np.asarray(cat_label)
        cat_label = cat_label.reshape((-1, window_size, n_classes))
    return cat_label
    
        

'''Hyperparameters'''
#DeepLearningModel
n_classes = 4
# PCA
do_PCA = False
do_balancing = True
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
        train_data, test_data = PCA_pipeline(train_data, test_data, stride)
    else:
        train_data, test_data= z_score_pipeline(train_data, test_data)
    if do_balancing:
      train_data = naive_balancer(train_data)
      #train_data = AVG_SMOTE(train_data)
     

    
    '''Deep Learning Model'''
    model =   keras.models.Sequential()
    model.add(Conv1D(128, kernel_size = 5, input_shape = (np.shape(train_data)[1], np.shape(train_data)[2] - 1), activation = 'relu', padding = 'valid'))
    #model.add(MaxPooling1D(pool_size=2, strides=2, padding = 'valid', data_format = 'channels_last'))
    model.add(Conv1D(128, kernel_size = 5, activation = 'relu', padding = 'valid'))
    #model.add(MaxPooling1D(pool_size=2, strides=3, padding = 'valid', data_format = 'channels_last'))
    model.add(Conv1D(128, kernel_size = 5, input_shape = (None, np.shape(train_data)[2]), activation = 'relu', padding = 'valid'))
    model.add(Conv1D(128, kernel_size = 5, input_shape = (None, np.shape(train_data)[2]), activation = 'relu', padding = 'valid'))
    model.add(Conv1D(128, kernel_size = 5, input_shape = (None, np.shape(train_data)[2]), activation = 'relu', padding = 'valid'))
    model.add(Conv1D(128, kernel_size = 4, input_shape = (None, np.shape(train_data)[2]), activation = 'relu', padding = 'valid'))
    model.add(Flatten())
    #model.add(LSTM(20, return_sequences=True))
    #model.add(LSTM(20, return_sequences=True))
#    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(n_classes, activation='softmax'))
    
#    '''training the model'''
#    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
#    history = model.fit(train_data[:,:,:-1], tf.keras.utils.to_categorical([democratic_Vote(train_data[i,:,-1]) for i in range(np.shape(train_data)[0])]), epochs = 10, validation_split = 0.2, batch_size = 100, shuffle = True)
#    
#    cont = input('do you wanna continue[Y],[N]')
#    while cont == 'Y':
#        history = model.fit(train_data[:,:,:-1], tf.keras.utils.to_categorical([democratic_Vote(train_data[i,:,-1]) for i in range(np.shape(train_data)[0])]), epochs = 10, validation_split = 0.2, batch_size = 100, shuffle = True)
#        cont = input('do you wanna continue[Y],[N]')
#    
#    score = model.evaluate(test_data[:,:,:-1], tf.keras.utils.to_categorical([democratic_Vote(test_data[i,:,-1]) for i in range(np.shape(test_data)[0])]))
    
    
    '''training the model'''
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    history = model.fit(train_data[:,:,:-1], one_hot_builder(train_data[:,:,-1], n_classes, window_size, mode = 'many_to_one'), epochs = 30, validation_split = 0.3, batch_size = 200, shuffle = True)
    
    cont = input('do you wanna continue[Y],[N]')
    while cont == 'Y':
        history = model.fit(train_data[:,:,:-1], one_hot_builder(train_data[:,:,-1], n_classes, window_size, mode = 'many_to_one'), epochs = 10, validation_split = 0.2, batch_size = 100, shuffle = True)
        cont = input('do you wanna continue[Y],[N]')
    
    score = model.evaluate(test_data[:,:,:-1], one_hot_builder(test_data[:,:,-1], n_classes, window_size, mode = 'many_to_one'))

    ''' summarize history for accuracy'''
    plt.figure(4)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    ''' summarize history for loss'''
    plt.figure(5)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()  










# import the preprocessed data
# kfold-crossvalidation
elif train_test_split == 'k_fold':
    fold_set = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\fold_set.npy')
    
