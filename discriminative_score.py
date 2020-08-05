# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:58:21 2020

@author: hartmann
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed, Conv1D
import matplotlib.pyplot as plt
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler




def random_generator(batch_size, window_length, mean, n_random, N):
    window_list = []
    dt = 1/ N
    for _ in range(batch_size):
        window = []
        vec = np.ones((n_random)) * mean
        for _ in range(window_length):
            for _ in range(N):
                vec = vec + np.random.normal(size =(n_random,)) * np.sqrt(dt) * 0.1
            window.append(vec)
        window_list.append(window)
    window_list = np.asarray(window_list)
    window_list = tf.convert_to_tensor(window_list, dtype = tf.float32)
        
    return window_list

def handle_missing_values(daten, method = 'linear_interpolation', threshhold_missingness = 0.5 ):
        '''method to deal with nan values'''
        # features with missingness of data above 50% are now deleted
        n_rows, n_cols = np.shape(daten)
#        for col in range(n_cols):
#            n_nans = np.sum(daten.iloc[:,col].isna())
#            percentage_nan = n_nans/n_rows
#            if  percentage_nan > threshhold_missingness:
#                daten.drop(columns = daten.columns[col], inplace = True)
        '''different methods to deal with missing data'''       
        if method == 'linear_interpolation':
            # interpolates the missing data in the sensorchannels linear over time with the subsequent values
            # if there are to much values missing subsequently then the missing values are filled with zeros
            daten = daten.interpolate(method='linear', limit_direction='forward', axis=0)
            daten = daten.fillna(0)
        elif method == 'MEAN':
            # imputes the missing values with the mean of the feature
            mean = np.mean(daten)
            n_rows, n_cols = np.shape(daten)
            for col in range (n_cols):
                nan_pos = np.where(daten.iloc[:,col].isna())[0]
                if len(nan_pos) != 0:
                    daten.iloc[nan_pos,col] = mean[col]
        elif method == 'last_carried_forward':
            # imputes values by carrying the last value forward eg. transcribing the missing value with the last observed value
            n_rows, n_cols = np.shape(daten)
            for col in range(n_cols):
                nan_pos = np.where(daten.iloc[:,col].isna())
                for row in nan_pos:
                    daten.iloc[row,col] = daten.iloc[row -1 , col]
                       
            daten = daten.fillna(0)
        return daten
            

def sliding(dataframe, window_size, stride, shuffle = False):
        '''puts a sliding window over a dataframe with
           :param: window size: kernel size of the sliding window
           :param: stride:      step size of the sliding window
           :boolean: shuffle:   shuffle the windows randomly for later machine learning algorithms'''
        n_windows = int((len(dataframe)-window_size)/stride)+1
        windowed_data = np.zeros((n_windows, window_size, np.shape(dataframe)[1]))
        
        for i in range(n_windows):
             windowed_data[i,:,:] = dataframe.iloc[i*stride:i*stride+window_size,:]
             
        if shuffle:
                np.random.shuffle(windowed_data)
                
        return windowed_data

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

def bce(y_label, y):
    loss = tf.math.reduce_mean(tf.losses.binary_crossentropy(y_label, y))
    return loss



def scaler(daten):
    scaler = MinMaxScaler()
    scaler.fit(daten)
    daten = pd.DataFrame(scaler.transform(daten))
    return daten

def myscaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    #data = np.asarray(data)    
    min_val = np.min(data, axis = 0)
    data = data - min_val
      
    max_val = np.max(data, axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data

'''opportunity data'''

split = 0.8

Data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')[:,:,:-1]
Data_labels = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')[:,:,-1]


'''Hyperparams'''
window_length = np.shape(Data)[1]
size_random = 60
n_samples = 1000
batch_size = 32


class Art_data:
    
    def __init__(self, window_length, size_random, n_samples):
        self.window_length = window_length
        self.size_random = size_random
        self.n_samples = n_samples
    
    def load_models(self, path):
        self.embedding = tf.keras.models.load_model(path + '\\embedding_weights')
        self.recovery = tf.keras.models.load_model(path + '\\recovery_weights')
        self.generator = tf.keras.models.load_model(path + '\\generator_weights')
        self.supervisor = tf.keras.models.load_model(path + '\\supervisor_weights')
#        self.embedding.summary()
#        self.recovery.summary()
#        self.generator.summary()
#        self.supervisor.summary()
    
    def random_generator(self, mean, N, n_samples):
        self.random_init = []
        dt = 1/ N
        for _ in range(n_samples):
            window = []
            vec = np.ones((self.size_random)) * mean
            for _ in range(self.window_length):
                for _ in range(N):
                    vec = vec + np.random.normal(size =(self.size_random,)) * np.sqrt(dt) * 0.1
                window.append(vec)
            self.random_init.append(window)
        self.random_init = np.asarray(self.random_init)
        self.random_init = tf.convert_to_tensor(self.random_init, dtype = tf.float32)
            


    '''creating new artificial data'''
    def create_art_data(self):
        #print(noise)
        E_hat = self.generator.predict(self.random_init)
        H_hat = self.supervisor.predict(E_hat)
        data = self.recovery.predict(H_hat)
        return data

path = 'C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights'

Synthesizer = Art_data(window_length, size_random, 1000)
Synthesizer.load_models(path)
Synthesizer.random_generator(0, 50, n_samples)
synthetic_data = Synthesizer.create_art_data()




def discriminative_score(real_data, syn_data):
    
    n_features = np.shape(real_data)[2]
    n_hidden = int (4 * n_features)
    n_epochs = 100
    batch_size = 32
    #n_batches = tf.shape(real_data)[0]/batch_size
    lr = 0.0001
    
    optimizer = tf.keras.optimizers.Adam(lr)
    
    real_data = tf.constant(real_data, dtype=tf.float32)
    real_data = tf.data.Dataset.from_tensor_slices(real_data)
    real_data = real_data.batch(batch_size, drop_remainder = True)
    
    syn_data = tf.constant(syn_data, dtype=tf.float32)
    syn_data = tf.data.Dataset.from_tensor_slices(syn_data)
    syn_data = syn_data.batch(batch_size, drop_remainder = True)
    
    
    disc = Sequential()
    disc.add(Bidirectional(LSTM(n_hidden, return_sequences = True, kernel_regularizer = regularizers.l2(0.01)), input_shape = (window_length, n_features)))
    disc.add(Bidirectional(LSTM(n_hidden, return_sequences = True, kernel_regularizer = regularizers.l2(0.01))))
    disc.add(Bidirectional(LSTM(n_hidden, return_sequences = False, kernel_regularizer = regularizers.l2(0.01))))
    disc.add(Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(0.01)))
    
    print('train discriminator model')
    
    for epoch in range(n_epochs):
        i = 0
        for batch_syn, batch_real in zip(syn_data, real_data):
            with tf.GradientTape() as tape:
                real_d = disc(batch_real)
                syn_d = disc(batch_syn)
                real_label = tf.zeros_like(batch_size)
                syn_label = tf.ones_like(batch_size)
                error = bce(real_label, real_d) + bce(syn_label, syn_d)
            disc_gradient = tape.gradient(error, disc.trainable_weights)
            optimizer.apply_gradients(zip(disc_gradient, disc.trainable_weights))
            print('disc_loss = ', error)
            i += 1
    
    #accuracy = disc.predict(test_daten)
            

discriminative_score(Data, synthetic_data)