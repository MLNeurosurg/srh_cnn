#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:35:27 2017
"""

# Importing images
import os
import numpy as np
import tensorflow as tf

# Keras Deep Learning modules
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Model and layer import
from keras.utils import multi_gpu_model

# Open-source models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121

# Model Layers
from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D

# Optimizers
from keras.optimizers import Adam

# Import callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau

# Sklearn modules
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score


##############################

# Image specifications/interpolation
training_dir = '/home/todd/Desktop/CNN_Images/nio_training_tiles'
validation_dir = '/home/todd/Desktop/CNN_Images/nio_validation_tiles'

img_rows = 300
img_cols = 300
img_channels = 3
    # Channel 0 = red [removed]
    ###########################
    # Channel 1 = Red/CH3-CH2
    # Channel 2 = Green/CH2
    # Channel 3 = Blue/CH3
total_classes = 14
class_names = ['ependymoma',
 'greymatter',
 'glioblastoma',
 'lowgradeglioma',
 'lymphoma',
 'medulloblastoma',
 'meningioma',
 'metastasis',
 'nondiagnostic',
 'normal',
 'pilocyticastrocytoma',
 'pituitaryadenoma',
 'pseudoprogression',
 'schwannoma',
 'whitematter']

def nio_preprocessing_function(image):
    """
    Channel-wise means calculated over NIO dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

def find_pair_factors_for_CNN(x):
    """
    Function to match batch size and iterations for the validation generator
    """
    pairs = []
    for i in range(2, 150):
        test = x/i
        if i * int(test) == x:
            pairs.append((i, int(test)))
    best_pair = pairs[-1]
    assert len(pairs) >= 1, "No pairs found"
    print(best_pair)
    return best_pair

########################### Image generators
train_generator = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function = nio_preprocessing_function,
    data_format = "channels_last").flow_from_directory(directory = training_dir,
    target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical',
    batch_size = 32, shuffle = True)
#    save_to_dir = "/home/orringer-lab/Desktop/keras_save_dir")

def validation_batch_steps(directory):
    counter = 0
    for roots, dirs, files in os.walk(directory):
        for file in files:
            counter += 1
    return find_pair_factors_for_CNN(counter)

val_batch, val_steps = validation_batch_steps(validation_dir)

validation_generator = ImageDataGenerator(
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function = nio_preprocessing_function,
    data_format = "channels_last").flow_from_directory(directory = validation_dir,
    target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical',
    batch_size = val_batch, shuffle = False)

################################

# MODEL BUILDING:
#base_model = InceptionV3(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3)) # must include the input shape to use keras-vis!!!
base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape = (img_rows, img_cols, 3))
#base_model = DenseNet121(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(.5)(x)
x = Dense(20, kernel_initializer='he_normal')(x)
x = BatchNormalization(name='todd_batch_norm')(x)
x = Activation("relu", name='todd_activation')(x)
x = Dense(total_classes, kernel_initializer='he_normal')(x)
predictions = Activation('softmax', name='todd_activation_2')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Distribute model across GPUs
parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.load_weights("/home/todd/Desktop/Final_Resnet_weights.06-0.86.hdf5")

# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

####### Define optimizer: ADAM
ADAM = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# COMPILE the model
#parallel_model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics =['accuracy'])
parallel_model.compile(optimizer=ADAM, loss="categorical_crossentropy", metrics =['accuracy'])

###############
# Callbacks/Early Stopping/ Model Monitoring
#early_stop = EarlyStopping(monitor='val_acc', min_delta = 0.05, patience=10, mode = 'auto')
checkpoint = ModelCheckpoint('Final_Resnet_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#tensorboard = TensorBoard(log_dir = "logs", histogram_freq=0, write_graph=False)
#reduce_LR = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=10, verbose=1, mode='auto', cooldown=0, min_lr=0)
callbacks_list = [checkpoint]

###############
# Class weights if you want them
class_weight = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
weight_dict = dict(zip(list(range(0,total_classes)), class_weight))

os.chdir('/home/todd/Desktop')
parallel_model.fit_generator(train_generator, steps_per_epoch = 10000, epochs=10, shuffle=True, class_weight = weight_dict,
                             max_queue_size=30, workers=1, initial_epoch=0, verbose = 1, validation_data=validation_generator, validation_steps=val_steps, callbacks=callbacks_list)


cnn_predictions = model.predict_generator(validation_generator, steps=val_steps, verbose=1)

# CNN prediction 1d vector. This is the inverse of one-hot encoding
cnn_predict_1d = np.argmax(cnn_predictions, axis = 1)

# Ground truth generated from the validation generator
index_validation = validation_generator.classes
# Overall accuracy
accuracy_score(index_validation, cnn_predict_1d)

def save_model(model, name):
    model.save(name + ".hdf5")

save_model(model, "transfertrain_model")
