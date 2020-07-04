 
import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




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
    

'''real data 4'''
filename = 'C:\\Users\\hartmann\\Desktop\\Gangdaten_timegan.csv'
daten = pd.read_csv(filename).iloc[:,2:]
daten = handle_missing_values(daten, method = 'linear_interpolation')
daten = myscaler(daten)
daten = sliding(daten, 32, 16)[0:20,:,:20]
Train_data = daten


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
    
    def random_generator(self, mean, N):
        self.random_init = []
        dt = 1/ N
        for _ in range(self.n_samples):
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
        noise = self.random_init
        #print(noise)
        E_hat = self.generator.predict(noise)
        H_hat = self.supervisor.predict(E_hat)
        data = self.recovery.predict(H_hat)
        return data

n_features = 20
path = 'C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights'
data_generator = Art_data(32, 20, 1000)
data_generator.load_models(path)
data_generator.random_generator(0, 50)
syn_data = data_generator.create_art_data()

mean_art_data = np.mean(syn_data, axis = 1)
mean_train_data = np.mean(Train_data, axis = 1)
std_art_data = np.std(syn_data, axis = 1)
std_train_data = np.std(Train_data, axis = 1)

for i in range(n_features):
    plt.figure(1)
    plt.subplot(4,5,i+1)
    plt.hist(mean_art_data[:, i], density=True)
    plt.hist(mean_train_data[:,i], density=True)
    plt.legend(['syn', 'real'])
    plt.xlabel('mean of feature')
    plt.ylabel('prob')
    plt.figure(2)
    plt.subplot(4,5,i+1)
    plt.hist(std_art_data[:, i], density=True)
    plt.hist(std_train_data[:,i], density=True)
    plt.legend(['syn', 'real'])
    plt.xlabel('std of feature')
    plt.ylabel('prob')




