#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse

import pydicom as dicom
import matplotlib.pyplot as plt
from pylab import rcParams  

from keras.models import load_model

from preprocessing.io import import_srh_dicom
from preprocessing.patch_generator import patch_generator
from prediction.prediction import feedforward, prediction, plot_srh_probability_histogram

parser = argparse.ArgumentParser(description="Use DeepSRH CNN to predict on raw SRH strips from a fresh surgical specimen")
parser.add_argument("-strip_dir", "--strip_directory", type=str, required=True, help="Path to directory that contains the raw SRH strips for surgical specimen.")
parser.add_argument("-model", "--model", type=str, required=True, help="Path to the trained CNN model for prediction. Should be saved as a keras .hdf5 model")
args = parser.parse_args()

CLASS_NAMES = ['ependymoma',
    'greymatter',
    'glioblastoma',
    'lowgradeglioma',
    'lymphoma',
    'medulloblastoma',
    'meningioma',
    'metastasis',
    'nondiagnostic',
    'pilocyticastrocytoma',
    'pituitaryadenoma',
    'pseudoprogression',
    'schwannoma',
    'whitematter']

if __name__ == "__main__":
    
    # load a trained SRH model
    deepSRHmodel = load_model(args.model)

    # import SRH mosaic
    specimen = import_srh_dicom(args.strip_directory)
    print("SRH mosaic size is: " + str(specimen.shape))

    # generate preprocessed image patches
    patches = patch_generator(specimen, step_size=100, renorm_red=True)
    del specimen

    # predict on patches
    specimen_prediction = prediction(feedforward(patch_dict = patches, model=deepSRHmodel))

    # print diagnosis
    print("SRH specimen diagnosis is: " + CLASS_NAMES[np.argmax(specimen_prediction)] + " (probability = " + str(np.max(specimen_prediction)) + ")")

    # display probability histogram
    plot_srh_probability_histogram(specimen_prediction)
    
