# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:34:12 2020

@author: hartmann
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:26:04 2020

@author: hartmann
"""

import keras 
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

def build_embedding(window_length, n_features, n_embedding):
    embedding = Sequential()
    embedding.add(LSTM(500, input_shape = (window_length, n_features), return_sequences = True))
    embedding.add(LSTM(500, return_sequences = True))
    embedding.add(LSTM(500, return_sequences = True))
    embedding.add(TimeDistributed(Dense(n_embedding, activation='sigmoid')))
    embedding.summary()
    return embedding

def build_reconstruction(window_length, n_embedding, n_features):
    reconst = Sequential()
    reconst.add(LSTM(500, input_shape = (window_length, n_embedding), return_sequences = True))
    reconst.add(LSTM(500, return_sequences = True))
    reconst.add(LSTM(500, return_sequences = True))
    reconst.add(TimeDistributed(Dense(n_features, activation = 'sigmoid')))
    reconst.summary()
    return reconst
    
#def build_reconstruction(window_length, n_embedding, n_features):
#    reconst = Sequential()
#    reconst.add(Dense(1000, input_shape = (window_length, n_embedding), activation = 'relu'))
#    reconst.add(Dense(1000, activation = 'relu'))
#    reconst.add(Dense(1000, activation = 'relu'))
#    reconst.add(Dense(n_features, activation = 'sigmoid'))
#    reconst.summary()
#    return reconst

def build_generator(window_length, size_random, n_embedding):
    generator = Sequential()
    #generator.add(Dense(300, input_shape = (size_random,)))
    #generator.add(RepeatVector(window_length))
    generator.add(LSTM(500, input_shape = (window_length, size_random), return_sequences = True, return_state = False))
    generator.add(LSTM(500, return_sequences = True, return_state = False))
    generator.add(LSTM(500, return_sequences = True, return_state = False))
#    generator.add(RepeatVector(window_length))
#    generator.add(LSTM(100, return_sequences = True, return_state = False))
#    generator.add(LSTM(100, return_sequences = True, return_state = False))
    #generator.add(TimeDistributed(Dense(20, activation='relu')))
    generator.add(TimeDistributed(Dense(n_embedding, activation = 'sigmoid')))
    generator.summary()
    return generator

def build_supervisor(window_length, n_embedding):
    supervisor = Sequential()
    supervisor.add(LSTM(500, input_shape = (window_length, n_embedding), return_sequences = True, return_state = False))
    supervisor.add(LSTM(500, return_sequences = True, return_state = False))
    supervisor.add(LSTM(500, return_sequences = True, return_state = False))
    supervisor.add(TimeDistributed(Dense(n_embedding, activation = 'sigmoid')))
    supervisor.summary()
    return supervisor


def build_discriminator(window_length, n_embedding):
    disc = Sequential()
    disc.add(Bidirectional(LSTM(500, return_sequences = True, kernel_regularizer = regularizers.l2(0.01)), input_shape = (window_length, n_embedding)))
    disc.add(Bidirectional(LSTM(500, return_sequences = True, kernel_regularizer = regularizers.l2(0.01))))
    disc.add(Bidirectional(LSTM(500, return_sequences = False, kernel_regularizer = regularizers.l2(0.01))))
    disc.add(Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(0.01)))
    disc.summary()
    return disc

def bce(y_label, y):
    loss = tf.math.reduce_mean(tf.losses.binary_crossentropy(y_label, y))
    return loss

def mse(x, x_hat):
    loss = tf.math.reduce_mean(tf.math.square(x-x_hat))
    #print('mseloss=', loss.numpy())
    return loss

def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)

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
    
'''Hyperparameters TimeGAN'''
n_embedding = 10
batch_size = 20
n_epochs = 100
size_random = 20
#Learning rates
lr2 = 0.0001
lr1 = 0.0001
#balancing loss
k = 1
mu = 10
gamma = 1

'''defining gradient descent optimizer'''
optimizer1 = tf.keras.optimizers.Adam(lr1)
optimizer2 = tf.keras.optimizers.Adam(lr2)

'''real Data 1'''
#train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')[0,:,:-1]
#Train_data = train_data.reshape((1,32,113))
#train_data_labels = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')[0,:,-1]
#vote = []
#for train_label in train_data_labels:
#    vote.append(democratic_Vote(train_label))
#(label_list, label_count) = np.unique(np.int32(vote), return_counts = True)
#add = np.max(label_count) - label_count
#
#
#window_length = np.shape(train_data)[1]
#n_features = np.shape(train_data)[2]
#n_samples = np.shape(train_data)[0]
#
#Train_data = train_data[ np.where(np.asarray(vote)==3),:,:].reshape(-1,window_length, n_features)

'''real data 2'''
#n= 1
#path = 'C:\\Users\\hartmann\\Desktop\\Jahn_smartphone_bewegungserkennung\\gehen/*.txt'
#alle_daten = []
#for filename in glob.glob(path):
#        daten = open(filename)
#        daten = daten.read()
#        daten = daten.strip()
#        daten = daten.split()
#        daten = daten[167+n*4 : -4 -n*4]            
#        daten = np.asarray(daten)
#        daten = np.float32(daten)
#        daten = np.reshape(daten,(-1,4))            # size 4: x axis y axis z axis time
#        if path == 'C:\\Users\\hartmann\\Desktop\\Jahn_smartphone_bewegungserkennung\\gehen/*.txt':
#            daten = np.concatenate((daten, np.zeros((len(daten[:,0]),1))),axis=1)
#        else:
#            daten = np.concatenate((daten, np.ones((len(daten[:,0]),1))),axis=1)
#        alle_daten.append(daten)
#alle_daten = np.vstack((alle_daten[0], alle_daten[1]))
#alle_daten = pd.DataFrame(alle_daten)
#alle_daten = scaler(alle_daten)
#
#Train_data = sliding(alle_daten, 128, 64)[0,:,:]
##Train_data = Train_data.reshape((1,128,-1))

'''real data 3'''
#filename = 'C:\\Users\\hartmann\\Desktop\\x_sense\\treppen_laufen\\treppe.txt'      
#daten = pd.read_csv(filename, encoding = "utf_16", sep='\t', decimal = ','  ).iloc[:, 2:20]
#daten = scaler(daten)
#Train_data = sliding(daten, 128, 64)[:, :, :]

'''real data 4'''
filename = 'C:\\Users\\hartmann\\Desktop\\Gangdaten_timegan.csv'
Train_data = pd.read_csv(filename).iloc[:,2:]
Train_data = handle_missing_values(Train_data, method = 'linear_interpolation')
Train_data = myscaler(Train_data)
Train_data = sliding(Train_data, 32, 16)[0:20,:,:20]




'''Making a toy dataset'''
#n_samples = 100
#Train_data = []
#for sample in range(n_samples):
#    window = np.linspace(-np.pi * np.linspace(0, 1, 100), np.pi * np.linspace(0, 1, 100), num = 32)
#    window = np.sin(window)/2 + 0.5
#    Train_data.append(window)
##randomscale = np.random.uniform(low = 0.1, high = 1, size = (n_samples))
#Train_data = np.asarray(Train_data)# + np.random.rand(100, 32, 100) / 10

'''Dimension of data'''
window_length = np.shape(Train_data)[1]
n_features = np.shape(Train_data)[2]
n_samples = np.shape(Train_data)[0]

'''Data pipeline for tensorflow'''
train_data = tf.constant(Train_data, dtype=tf.float32)
train_data = tf.data.Dataset.from_tensor_slices(train_data)
train_data = train_data.batch(batch_size, drop_remainder = True)

'''building embedding reconstruction generator and discriminator'''
embedding = build_embedding(window_length, n_features, n_embedding)
reconst = build_reconstruction(window_length, n_embedding, n_features)
generator = build_generator(window_length, size_random, n_embedding)
disc = build_discriminator(window_length, n_embedding)
supervisor = build_supervisor(window_length, n_embedding)


'''beginn custom training loop'''

print('pretraining embedding network...')
for epoch in range(int(n_epochs * 20)):
    for X in train_data:
        with tf.GradientTape(watch_accessed_variables = True, persistent = True) as tape: 
            H = embedding(X)
            X_tilde = reconst(H)
                
            E_loss_T0 = mse(X, X_tilde)
            E_loss0 = 10* tf.sqrt(E_loss_T0)
        e_grad = tape.gradient(E_loss0, embedding.trainable_variables)
        r_grad = tape.gradient(E_loss0, reconst.trainable_variables)
        optimizer1.apply_gradients(zip(e_grad, embedding.trainable_variables))
        optimizer1.apply_gradients(zip(r_grad, reconst.trainable_variables))
        if epoch%10==0:
            print(epoch)
            print(E_loss0)
print('finished pretraining embedding network')

print('training generator with supervised loss...')
for epoch in range(int(n_epochs * 12)):
    for X in train_data:
        with tf.GradientTape() as tape:
            H = embedding(X)
            noise =  random_generator(batch_size, window_length, 0, size_random, 50)
            E_hat = generator(noise)
            H_hat = supervisor(E_hat)
            H_hat_supervise = supervisor(H)
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
        s_grad = tape.gradient(G_loss_S, supervisor.trainable_variables)
        optimizer1.apply_gradients(zip(s_grad, supervisor.trainable_variables))
        if epoch%10==0:
            print(epoch)
            print(G_loss_S)
print('finished training with supervised loss')

#print('pretrain discriminator')
#for i in range(50):
#    for batch in train_data:
#        with tf.GradientTape(watch_accessed_variables = True, persistent = True) as tape:
#            latent_features = embedding(batch)
#            reconst_features = reconst(latent_features)
#            
#            y = disc(latent_features)
#            yhat = disc(generated_latent)
#            y_label = tf.zeros_like(y)
#            yhat_label = tf.ones_like(yhat)
#            
#            
#            
#            Lu_disc = bce(y_label, y)
#            Lu_disc_gen = bce(yhat_label, yhat) 
#            
#            disc_loss = (Lu_disc + Lu_disc)/2
#            
#        d_grad =       tape.gradient(disc_loss, disc.trainable_variables)
#        optimizer2.apply_gradients(zip(d_grad, disc.trainable_weights))
#       
#        print(disc_loss)
       
print('finish pretrtain disc')       
        
        

history_discriminator = []
history_generator = []
print('joint training...')
for epoch in range (n_epochs*10):
    print('epoch:',epoch)
    for X in train_data:
        for kk in range(2):
            with tf.GradientTape(watch_accessed_variables = True, persistent = True) as tape:
                H = embedding(X)
                X_tilde = reconst(H)
                
                noise =  random_generator(batch_size, window_length, 0, size_random, 50)
                E_hat = generator(noise)
                H_hat = supervisor(E_hat)
                H_hat_supervise = supervisor(H)
                    
                X_hat = reconst(H_hat)
                
                
                Y_fake = disc(H_hat)
                Y_real = disc(H)
                Y_fake_e = disc(E_hat)
                                
                                
                D_loss_real = bce(tf.ones_like(Y_real), Y_real)
                D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake)
                D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e)   
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
                                
                G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)   
                G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)
                
                G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                
                G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
                G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
                
                G_loss_V = G_loss_V1 + G_loss_V2
                
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
                
                E_loss_T0 = mse(X, X_tilde)
                E_loss0 = 10* tf.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * G_loss_S
                                
            d_grad = tape.gradient(D_loss, disc.trainable_weights)
            optimizer1.apply_gradients(zip(d_grad, disc.trainable_weights))
        e_grad = tape.gradient(E_loss, embedding.trainable_weights)
        optimizer1.apply_gradients(zip(e_grad, embedding.trainable_weights))
        g_grad = tape.gradient(G_loss, generator.trainable_weights) 
        optimizer1.apply_gradients(zip(g_grad, generator.trainable_weights)) 
        s_grad = tape.gradient(G_loss, supervisor.trainable_weights) 
        optimizer1.apply_gradients(zip(s_grad, supervisor.trainable_weights))               
        history_generator.append(G_loss)
        history_discriminator.append(D_loss)
                                
        
           
            
    if epoch % 2 ==0:
        print(" loss disc = {}\nloss gen = {}\nloss embedding = {}".format(D_loss, G_loss, E_loss ))
    
    if epoch % 10 == 0:
        print('saving model weights')
        embedding.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights\\embedding_weights')
        reconst.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights\\recovery_weights')
        generator.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights\\generator_weights')
        supervisor.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights\\supervisor_weights')
        disc.save('C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights\\discriminator_weights')
        print('finished saving model weights')

plt.figure(100)
plt.plot(history_generator)
plt.plot(history_discriminator)
       
print('finished joint training')    

'''creating new artificial data'''
def create_art_data(generator, supervisor, reconstructor, n_samples, window_length, size_random):
    noise = random_generator(n_samples, window_length, 0, size_random, 50)
    #print(noise)
    E_hat = generator.predict(noise)
    H_hat = supervisor.predict(E_hat)
    data = reconstructor.predict(H_hat)
    return data

art_data = create_art_data(generator, supervisor, reconst, 100, 32, size_random)
for sample in range(3):
    for k in range(5):
        #plt.figure(sample)
        plt.plot(art_data[sample,:,k], color='red')
        plt.plot(Train_data[sample,:, k], color = 'blue')
        plt.legend(['generated', 'Toy-time series'])
        

mean_art_data = np.mean(art_data, axis = 1)
mean_train_data = np.mean(Train_data, axis = 1)
std_art_data = np.std(art_data, axis = 1)
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
    
path = 'C:\\Users\\hartmann\\Desktop\\Opportunity\\Timegan_weights'

embedding = tf.keras.models.load_model(path + '\\embedding_weights')
recovery = tf.keras.models.load_model(path + '\\recovery_weights')
generator = tf.keras.models.load_model(path + '\\generator_weights')
supervisor = tf.keras.models.load_model(path + '\\supervisor_weights')
disc = tf.keras.models.load_model(path + '\\embedding_weights')
