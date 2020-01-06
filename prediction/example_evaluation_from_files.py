#!/usr/bin/env python3

import os
import numpy as np

# Keras Deep Learning modules
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import backend as K

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from training.model_training import validation_batch_steps
from prediction.evaluation_from_files import *


def cnn_predictions(model, validation_generator):
    cnn_predictions = model.predict_generator(validation_generator, steps = val_steps, verbose = True)
    return cnn_predictions

# Import model
parallel_model = load_model('')

# Specify directory with images for prediction
validation_dir = ''

# Define batch size and number of steps
val_batch, val_steps = validation_batch_steps(validation_dir)

# instantiate validation generator
validation_generator = ImageDataGenerator(
    samplewise_center=False,
    samplewise_std_normalization = False,
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function=nio_preprocessing_function,
    data_format = "channels_last").flow_from_directory(directory = validation_dir,
    target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical',
    batch_size=val_batch, shuffle = False)

# model predictions
preds = cnn_predictions(parallel_model, validation_generator)

# instantiate model object
model_object = TrainedCnnModel(parallel_model, validation_generator, preds)

# evaluate model performance
model_object.mosaic_softmax_all_normal_filter()
model_object.patient_softmax_all_normal_filter()

# multiclass confusion matrices
tile_level_multiclass_confusion_matrix(model_object, normalize=True)
mosaic_level_multiclass_confusion_matrix(model_object, normalize=True)
patient_level_multiclass_confusion_matrix(model_object, normalize=True)




