#!/usr/bin/env python3

'''
Script that: 
1) generates patches from strip directories
2) saves them to single specified directory

Use patch_generator_model_filter.py to use the trained model to filter images into seperate directories
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

import pydicom as dicom

from preprocessing.preprocess import *
from preprocessing.registration import *
from preprocessing.io import import_preproc_dicom, import_raw_dicom

IMAGE_SIZE = 300

def starts_finder(side, stride, image_size): 
    """
    Helper function used in patch_generator function to find maximum number of patches that can be obtained starting with the upper left hand pixel as index 0. 
    We do NOT use padding for our prediction or heatmaps, so the right edge of the image may be excluded from prediction/segmentation. 
    side = mosaic image side (height or width)
    stride = step size for sliding window algorithm
    image_size = patch image size (height or width)
    """
    starts = [0] 
    # loop until off the image
    while (starts[-1] + stride + image_size) <= side:
        starts.append(starts[-1] + stride)
    return starts

def patch_generator(preprocessed_mosaic, step_size = 100, old_preprocess = False):
    """
    Function that accepted preprocessed mosaic 
    """
    starts = starts_finder(side = preprocessed_mosaic.shape[0], stride = step_size, image_size=IMAGE_SIZE)
    patch_dict = {}
    counter1 = 0

    # loop through the x and y axes
    for y in starts:
        for x in starts:
            # select the region of the mosaic
            patch = preprocessed_mosaic[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE, :]
            _, CH2, CH3 = return_channels(patch)
            CH2 = min_max_rescaling(CH2)
            CH3 = min_max_rescaling(CH3)
            
            # channel subtraction
            subtracted_array = np.subtract(CH3, CH2) # CH3 minus CH2
            subtracted_array[subtracted_array < 0] = 0.0 # negative values set to zero

            if old_preprocess:
                subtracted_array = min_max_rescaling(subtracted_array)

            # concatentate the postprocessed images
            stack = np.zeros((CH2.shape[0], CH2.shape[1], 3), dtype=np.float)
            stack[:, :, 0] = subtracted_array
            stack[:, :, 1] = CH2
            stack[:, :, 2] = CH3
            patch_dict[counter1] = stack * 255
            counter1 += 1

    return patch_dict
    
def patch_saver(patch_dict, src_dir):

    # patch counters
    saved = 0
    mosaic_num = src_dir.split("/")[-1]

    # iterate and predict over the CNN patches
    for index, image in patch_dict.items():
        imsave(mosaic_num + "_" + str(index) + ".tif", patch_dict[index].astype(np.uint8))	
        print(mosaic_num + "_" + str(index) + ".tif")
        saved += 1

    print("Total saved = " + str(saved))

def directory_list(parent_dir):
    """
    Function to return a list of directories that contain raw SRH strips 
    """
    dir_list = sorted(os.listdir(parent_dir))

    directory_list = []
    for directory in dir_list:
        directory_list.append(os.path.join(parent_dir, directory))

    return directory_list


if __name__ == "__main__":
    
    # root directory with directories of STRIPS from single mosaic
    src_dir = "/home/todd/Desktop/test_strips"
    # destination for ALL PATCHES
    dest_dir = "/home/todd/Desktop/test_patch_generator"

    # ordered list of strip directories
    ordered_dir_list = directory_list(src_dir)

    for strip_directory in ordered_dir_list:
        print(strip_directory)
        # import mosaic
        mosaic = import_preproc_dicom(strip_directory, flatten = True, filter_type = "gaussian")
        # returns the patches to save and patches to pass to CNN
        patches = patch_generator(mosaic, step_size=200)
        # save patches
        os.chdir(dest_dir)
        patch_saver(patches, src_dir = strip_directory)


