#!/usr/bin/env python3

'''
Script that 
1) generates patches from strip directories, 
2) sorts them using trained model into diagnostic, nondiagnostic, normal
3) saves them to specified directories
'''

# standard python
import os
import sys
from collections import defaultdict, OrderedDict
# data science
import numpy as np
import pandas as pd
from scipy import stats
# plotting
import matplotlib.pyplot as plt
import pydicom as dicom
import argparse
from imageio import imsave

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.utils import multi_gpu_model
from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D 

import pydicom as dicom

from preprocessing.preprocess import *
from preprocessing.registration import *
from preprocessing.io import import_srh_dicom
from preprocessing.patch_generator import patch_generator, starts_finder

IMAGE_SIZE = 300

def import_filtering_model(model_path, gpu_number = 1):

    """
    Imports a keras model that was saved as weights only
    """
    img_rows = 300
    img_cols = 300
    total_classes = 14
    base_model = InceptionResNetV2(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(.5)(x)  
    x = Dense(20, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(total_classes, kernel_initializer='he_normal')(x)
    predictions = Activation('softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    if gpu_number:
        
        parallel_model = multi_gpu_model(model, gpus=gpu_number)
        
        return parallel_model
    else:
        parallel_model.load_weights(model_path)
    return parallel_model

def patch_generator_cnn_filter(preprocessed_mosaic, step_size = 200):
    """
    Returns a tuple containing a dictionary of regular patches and a dictionary of patches for CNN prediction. 
    This function is needed because some elements of our preprocessing pipeline has changed. 
    """
    starts = starts_finder(side = preprocessed_mosaic.shape[0], stride = step_size, image_size = IMAGE_SIZE)
    cnn_patches = {}
    counter = 0

    # images for CNN ONLY
    for y in starts:
        for x in starts:
            patch = preprocessed_mosaic[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE, :]
            patch = channel_rescaling(patch)
            cnn_patches[counter] = patch
            counter += 1
            print(counter)

    patch_dict = patch_generator(preprocessed_mosaic, step_size = step_size)

    return patch_dict, cnn_patches
    
def patch_filter_saver(patch_dict, cnn_patch_dict, src_dir, model = None):

    # patch counters
    nondiagnostic = 0
    diagnostic = 0
    normal = 0
    mosaic_num = src_dir.split("/")[-1]

    # iterate and predict over the CNN patches
    for index, image in cnn_patch_dict.items():

        try:
            # predict on image
            pred_image = np.expand_dims(cnn_preprocessing(image.astype(float)), axis = 0)
            pred = model.predict(pred_image, batch_size = 1, verbose = False)
            
            # Filtered based on prediction
            # NONDIAGNOSTIC IMAGE   
            if np.argmax(pred) == 8:     
                print("nondiagnostic patch")
                nondiagnostic += 1
                os.chdir(NONDIAGNOSTIC_DIR)
                imsave(mosaic_num + "_" + str(index) + ".tif", patch_dict[index].astype(np.uint8))
                continue

            # NORMAL IMAGE
            elif (np.argmax(pred) == 2) or (np.argmax(pred) == 13) or (np.argmax(pred) == 11):
                print("normal patch")	
                normal += 1            
                os.chdir(NORMAL_DIR)
                imsave(mosaic_num + "_" + str(index) + ".tif", patch_dict[index].astype(np.uint8))
                continue 

            # DIAGNOSTIC IMAGE
            else:
                print("diagnostic patch")
                diagnostic += 1
                os.chdir(DIAGNOSTIC_DIR)
                imsave(mosaic_num + "_" + str(index) + ".tif", patch_dict[index].astype(np.uint8))	
        
        except:
            continue	

    total_images = nondiagnostic + diagnostic + normal
    print("% Normal: " + str(np.round((normal / total_images) * 100, decimals = 3)))
    print("% Nondiagnostic: " + str(np.round((nondiagnostic / total_images) * 100, decimals = 3)))
    print("% Diagnostic: " + str(np.round((diagnostic / total_images) * 100, decimals = 3)))

def directory_list(parent_dir):
    """
    Function to return a channel_rescaling
    """

    dir_list = sorted(os.listdir(parent_dir))

    directory_list = []
    for directory in dir_list:
        directory_list.append(os.path.join(parent_dir, directory))

    return directory_list


if __name__ == "__main__":
    

    # root directory with directories of STRIPS from single mosaic
    src_dir = "/home/todd/Desktop/test_strips"

    # specify directorties for normal and nondiagnostic
    DIAGNOSTIC_DIR = ""
    NORMAL_DIR = ""
    NONDIAGNOSTIC_DIR = ""

    # import filtering model
    model_path = "/home/todd/Desktop/Models/Final_Resnet_weights.03-0.86PAPERMODEL.hdf5"
    filtering_model = import_filtering_model(model_path = model_path)

    # ordered list of strip directories
    ordered_dir_list = directory_list(src_dir)

    for strip_directory in ordered_dir_list:
        print(strip_directory)
        # import mosaic
        mosaic = import_srh_dicom(strip_directory, flatten = False)
        # returns the patches to save and patches to pass to CNN
        patches, cnn_patch_dict = patch_generator_cnn_filter(mosaic, step_size=200)
        # save patches
        patch_filter_saver(patches, cnn_patch_dict, src_dir = strip_directory, model = filtering_model)



