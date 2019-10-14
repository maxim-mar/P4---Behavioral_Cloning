# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:19:14 2019

@author: ZZ1LRN
"""

'''
In order to ensure that the neural network can be trained properly,
several preprocessing steps are required. All the prerocessing activities will be done within this file.
'''

import numpy as np
import pandas as pd
from keras import models, optimizers, backend
from keras.models import Sequential
from keras.layers import Conv2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, Activation, Cropping2D
from keras.preprocessing.image import img_to_array, load_img
import cv2
import os
from keras.layers.normalization import BatchNormalization


#Steeering correction factor
steering_correction = 0.25

#Batch Size
batch_size=40

#Number of Epochs
EPOCHS = 3

def get_augmented_row(row):
    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += steering_correction
    elif camera == 'right':
        steering -= steering_correction

    #Load the image
    curr_path = os.getcwd()
    image = load_img(curr_path + row[camera].strip())
    image = img_to_array(image)
    

    # decide randomly if the image should be fillped horizontaly
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    return image, steering


def get_data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def get_model():
    model = Sequential()

    # Normalization and coping of the Input image
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,30), (0,0)), input_shape=(160,320,3)))
    
    
    # layer 1 - Convolution layer with 'relu' activation
    # Output: 32x32x32
    model.add(Conv2D(24, 5, 5, input_shape=(80, 320, 3), subsample=(2, 2), border_mode="valid"))
    #Layer Activation
    model.add(Activation('relu'))


    # layer 2 - Convolution layer with dropout, 'relu' activation & maxPooling
    # Output: 30x30x16
    model.add(Conv2D(36, 5, 5,  subsample=(2,2), border_mode="valid"))
    #Layer Activation
    model.add(Activation('relu'))
    #Apply MaxPooling
    #Output 15x15x16
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    
    
    # layer 3 - Convolution layer with dropout and 'relu' activation
    # Output 13x13x16
    model.add(Conv2D(48, 3, 3, border_mode="valid"))
    #Layer Activation
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())


    # layer 3 - Convolution layer with dropout and 'relu' activation
    # Output 13x13x16
    model.add(Conv2D(64, 3, 3, border_mode="valid"))
    #Layer Activation
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Flatten the output
    model.add(Flatten())

    ## layer 4 - - Fully Connected Layer with dropout an 'relu' activation
    model.add(Dense(1164))
    #Layer Activation
    model.add(Activation('relu'))
    #Apply Dropout
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    ## layer 5 - Fully Connected Layer with dropout an 'relu' activation
    model.add(Dense(100))
    #Layer Activation
    model.add(Activation('relu'))
    #Apply Dropout
    #model.add(Dropout(0.5))
    model.add(BatchNormalization())

    ## layer 6 - Fully Connected Layer with 'relu' activation
    model.add(Dense(50))
    model.add(Activation('relu'))
    #Apply Dropout
   # model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # Final Layer with single output
    model.add(Dense(1))

    #Apply Adam-Optimizer and MSE loss fucntion
    model.compile(optimizer="adam", loss="mse")

    return model

if __name__ == "__main__":
    BATCH_SIZE = batch_size

    data_frame = pd.read_csv('./data/driving_log.csv', usecols=[0, 1, 2, 3])
    data_frame.columns = ['center','left', 'right', 'steering']

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)


    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    
    # release the main data_frame from memory
    data_frame = None

    training_generator = get_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(validation_data, batch_size=BATCH_SIZE)

    model = get_model()
    
    #Show the summary of all layers of the model
    print((model.summary()))


    steps_per_epoch = ((len(training_data)*6)//BATCH_SIZE)

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        steps_per_epoch=steps_per_epoch, nb_epoch=EPOCHS, validation_steps=(len(validation_data)))

    print("Saving model weights and configuration file.")    
    model.save('model.h5')
