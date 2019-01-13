#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 00:38:23 2018

@author: orringer-lab
"""


# Importing images
import os
import random
import numpy as np
from skimage.io import imread 
#from preprocessing import srh_preprocessing
# Image manipulation

# Keras Deep Learning modules
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator # may not need this
# Model and layer import
from keras.utils import multi_gpu_model

# Open-source models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet169

from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D 
from keras.regularizers import l2 

# Optimizers
from keras.optimizers import Adam, SGD, RMSprop


training_dir = '/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/training_nondiag_diag'
validation_dir = '/home/orringer-lab/Desktop/CNN_DiagNondiag/validation'

# Image specifications/interpolation
img_rows = 224
img_cols = 224
img_channels = 3

#def find_pair_factors_for_CNN(validation_generator):
#    """
#    Function to match batch size and iterations for the validation generator
#    """
#    assert type(validation_generator.n) == int, "x is not an integer."
#    x = validation_generator.n
#    pairs = []
#    for i in range(2, 60):
#        test = x/i
#        if i * int(test) == x:
#            pairs.append((i, int(test)))
#    best_pair = pairs[-1]
#    return best_pair

train_generator = ImageDataGenerator(
    samplewise_center=False, 
    samplewise_std_normalization = False,
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range = 0.,
    data_format = "channels_last").flow_from_directory(directory = training_dir, 
    target_size = (img_rows, img_cols), interpolation='bicubic', color_mode = 'rgb', classes = None, class_mode = 'categorical', 
    batch_size = 24, shuffle = True)
#    save_to_dir = "/home/orringer-lab/Desktop/keras_save_dir")

def find_pair_factors_for_CNN(x):
    """
    Function to match batch size and iterations for the validation generator
    """
    assert type(x) == int, "x is not an integer."
    pairs = []
    for i in range(2, 30):
        test = x/i
        if i * int(test) == x:
            pairs.append((i, int(test)))
    best_pair = pairs[-1]
    return best_pair

def validation_batch_steps(directory):
    counter = 0
    for roots, dirs, files in os.walk(directory):
        for file in files:
            counter += 1
    return find_pair_factors_for_CNN(counter)

val_batch, val_steps = validation_batch_steps(validation_dir)

validation_generator = ImageDataGenerator(
    samplewise_center=False, 
    samplewise_std_normalization = False,
    horizontal_flip=False,
    vertical_flip=False,
    data_format = "channels_last").flow_from_directory(directory = validation_dir,
    target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical', 
    batch_size = 14, shuffle = False)

base_model = DenseNet169(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(.4)(x) # Dropout layer in same position as InceptionV3 architecture
predictions = Dense(2, activation='softmax')(x) 
model = Model(inputs=base_model.input, outputs=predictions)

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

####### ADAM (RMSprop and momentum)
#ADAM = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.0005, momentum=0.9, nesterov=True)  

# COMPILE the model
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics =['accuracy'])

###############
# Class weights if you want them
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
weight_dict = dict(zip(list(range(0,13)), class_weight))

#def execute_training():
os.chdir('/home/orringer-lab/Desktop')
model.fit_generator(train_generator, steps_per_epoch = 3088, epochs=1, shuffle=True, class_weight = weight_dict, 
                             validation_data=validation_generator, validation_steps=343, max_queue_size=10, workers=1, initial_epoch=0, verbose = 1)


def save_model(model, name):
    model.save(name + ".hdf5")

save_model(model, "DiagNondiag_model_04222018_acc99_longertrain")


 
