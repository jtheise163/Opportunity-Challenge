
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:44:45 2020

@author: hartmann
"""

#import numpy as np
#import tensorflow as tf
#from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
#from tensorflow.keras.layers import Flatten, BatchNormalization
#from tensorflow.keras.layers import Activation, ZeroPadding2D
#from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.models import Sequential, Model, load_model
#from tensorflow.keras.optimizers import Adam
#
#def democratic_Vote(labels):
#    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
#    index = np.where(np.max(label_count))[0]
#    vote = label[index] 
#    vote = int(vote)
#    return vote
#
#def z_score(data):
#    '''computes z-scores of data 
#       every feature has a mean of 0 and a standard deviation of 1
#       :param:   data:   2-d Numpy array: datapoint x feature
#       :returns: z-score:2-d Numpy array: datapoint x feature (mean = 0, std=1) '''
#    mean = np.mean(data, axis = 0) #columnswise mean and standard_deviation
#    std = np.std(data, axis = 0)
#    data = (data - mean)/std
#    data = np.nan_to_num(data)
#    return data
#
#
#def z_score_pipeline(train_data, test_data):
#    train_data_z = [] 
#    counter = 0
#    for window in train_data[:,:,:-1]:
#        window = z_score(window)
#        window = np.concatenate((window, train_data[counter, :, -1].reshape((-1,1))), axis = 1)
#        train_data_z.append(window)
#        counter += 1
#    train_data_z = np.asarray(train_data_z)
#    
##    test_data_z = [] 
##    counter = 0
##    for window in test_data[:,:,:-1]:
##        window = z_score(window)
##        window = np.concatenate((window, test_data[counter, :, -1].reshape((-1,1))), axis = 1)
##        test_data_z.append(window)
##        counter += 1
##    test_data_z = np.asarray(test_data_z)
#    return train_data_z#
#
#
#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
#def discriminator_loss(real_output, fake_output):
#    real_loss = cross_entropy(tf.ones_like (real_output), real_output)
#    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#    return real_loss + fake_loss
#
#def generator_loss(fake_output):
#    return cross_entropy(tf.ones_like(fake_output), fake_output)
#
#def build_generator(seed_size, data_shape):
#    model = Sequential()
#    model.add(Dense(10, activation = 'relu', input_dim = seed_size))
#    model.add(LSTM(128, return_sequences=True))
#    #model.add(BatchNormalization(momentum=0.8))
#    model.add(LSTM(data_shape[1]-1,  return_sequences=False))
#    return model
#
#def build_discriminator(data_shape):
#    model = Sequential()
#    model.add(LSTM(128, input_shape = data_shape, return_sequences=True))
#    #model.add(BatchNormalization(momentum=0.8))
#    model.add(LSTM(128, return_sequences=False))
#    #model.add(BatchNormalization(momentum=0.8))
#    model.add(Dense(1, activation = 'sigmoid'))
#    return model
#    
#    
#generator_optimizer     = tf.keras.optimizers.Adam(1.5e-4, 0.5)
#discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
#
#
#EPOCHS = 50
##BATCH_SIZE = 32
#SEED_SIZE = 100
#BUFFER_SIZE = 60000
#
#train_data = np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy')
##train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#data_shape = (np.shape(train_data)[1], np.shape(train_data)[2]-1)
#generator = build_generator(SEED_SIZE, data_shape)
#discriminator = build_discriminator(data_shape)
#
#@tf.function
#def train_step(batch):
#    #seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])
#    seed = tf.random.normal([SEED_SIZE])
#    
#    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#        generated_series = generator(seed, training = True)
#        
#        real_output = discriminator(batch, training = True)
#        fake_output = discriminator(generated_series, training = True)
#        
#        gen_loss = generator_loss(fake_output)
#        disc_loss = discriminator_loss(real_output, fake_output)
#        
#        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#        
#        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#    return gen_loss, disc_loss
#
#
#def train(train_data, epochs):
#    
#    for epoch in range(epochs):
#        gen_loss_list = []
#        disc_loss_list = []
#        for window in train_data:
#            t = train_step(window)
#            gen_loss_list.append(t[0])
#            disc_loss_list.append(t[1])
#            
#            g_loss = sum(gen_loss_list)/len(gen_loss_list)
#            d_loss = sum(disc_loss_list)/len(disc_loss_list)
#            
#            print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
#               ' {hms_string(epoch_elapsed)}')
#    
#
#train(train_data, EPOCHS)
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K 
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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

def z_score_pipeline(train_data):
    train_data_z = [] 
    counter = 0
    for window in train_data[:,:,:-1]:
        window = z_score(window)
        window = np.concatenate((window, train_data[counter, :, -1].reshape((-1,1))), axis = 1)
        train_data_z.append(window)
        counter += 1
    train_data_z = np.asarray(train_data_z)
    return train_data_z

def democratic_Vote(labels):
    (label, label_count) = np.unique(np.int32(labels), return_counts = True)
    index = np.where(np.max(label_count))[0]
    vote = label[index] 
    vote = int(vote)
    return vote

def build_generator(inp_shape, output_shape):
    model =  keras.models.Sequential()
    model.add(LSTM(128, input_shape = inp_shape, return_sequences=True))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_shape, activation='linear'))
    return model

def build_discriminator(inp_shape):
    model =  keras.models.Sequential()
    model.add(LSTM(128, input_shape = inp_shape, return_sequences=True))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(128, return_sequences=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1, activation = 'sigmoid'))
    return model


def create_batch(data, batch_size):
    n_samples = int(len(data) / batch_size)
    batch_list = []
    for i in range(n_samples):
        batch_list.append(data[i*batch_size:(i+1)*batch_size,:,:])
    return batch_list
        

def build_gan(generator, discriminator, z):
    gan_core = discriminator(generator(z))
    gan = keras.models.Model(inputs = z, outputs=gan_core)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

K.clear_session()


train_data_with_label = z_score_pipeline(np.load('C:\\Users\\hartmann\\Desktop\\Opportunity\\processed_data\\train_data.npy'))
train_data = train_data_with_label[:,:,:-1]
train_label_list = train_data_with_label[:,:,-1]
del train_data_with_label

vote = []
for train_label in train_label_list:
    vote.append(democratic_Vote(train_label))
(label_list, label_count) = np.unique(np.int32(vote), return_counts = True)
add = np.max(label_count) - label_count



EPOCHS = 4000
RANDOM_SIZE = 100
N_SENSORS = np.shape(train_data)[2]
WINDOW_LENGTH = np.shape(train_data)[1]
n_samples = np.shape(train_data)[0]

train_data = train_data[ np.where(np.asarray(vote)==3),:,:].reshape(-1,WINDOW_LENGTH, N_SENSORS)


input_shape_generator = (WINDOW_LENGTH, RANDOM_SIZE)
input_shape_discriminator = (WINDOW_LENGTH, N_SENSORS)
output_shape = N_SENSORS

#train_data = create_batch(train_data, 100)
n_samples = np.shape(train_data)[0]
random = np.random.rand(n_samples, WINDOW_LENGTH, RANDOM_SIZE)


generator = build_generator(input_shape_generator, output_shape)
generator.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer='adam')
discriminator = build_discriminator(input_shape_discriminator)
discriminator.compile(loss='binary_crossentropy',optimizer='adam')
z = keras.layers.Input(shape=(input_shape_generator))
gan = build_gan(generator, discriminator, z)

seeded_random = np.random.rand(1, WINDOW_LENGTH, RANDOM_SIZE)
disc_loss_history = []
gen_loss_history = []
for epoch in range(EPOCHS):
    print('Epoch:',epoch)
    random = np.random.rand(n_samples, WINDOW_LENGTH, RANDOM_SIZE)
    generated_series = generator.predict(random)
    data = np.vstack((generated_series, train_data))
    label = np.int32([i<n_samples for i in range(n_samples*2)])
    data, label = shuffle(data, label, random_state=0)
    print('discriminator training...')
    disc_loss = discriminator.train_on_batch(data, label)
    print(disc_loss)
    disc_loss_history.append(disc_loss)
    discriminator.trainable = False
    #gan = build_gan(generator, discriminator, z)
    print('generator training...')
    gen_loss = gan.train_on_batch(random, np.zeros(n_samples))
    print(gen_loss)
    gen_loss_history.append(gen_loss)
    discriminator.trainable = True
#    if epoch % 10 == 0:
#        example = generator.predict(seeded_random)
        #plt.plot(example[0,:,100])
    
    
    
plt.figure(2)
plt.plot(gen_loss_history)
plt.plot(disc_loss_history)





































































