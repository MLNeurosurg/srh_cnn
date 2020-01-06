#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from keras.models import load_model
from pandas import DataFrame
from collections import defaultdict
from pylab import rcParams  

from preprocessing.preprocess import *
from preprocessing.patch_generator import patch_generator
from preprocessing.io import import_srh_dicom

# class list
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

def feedforward(patch_dict, model):
    """
    Function to perform a forward pass on all patches in a patch dictionary
    Note: There is NO filtering of the nondiagostic patches with this function. 
    """
    num_classes = model.output_shape[-1] 

    softmax_output = np.zeros((1,num_classes), dtype=float)
    nondiag_count = 0
    
    print("Total patches: " + str(len(patch_dict)))
    for key, patch in patch_dict.items():
        plt.imshow(patch.astype(np.uint8))
        plt.show()
        patch = cnn_preprocessing(patch)
        pred = model.predict(patch[None,:,:,:], batch_size = 1)
        if np.argmax(pred) == 8:
            continue
        else:
            softmax_output += pred # unnormalized probabability distribution

        if (key % 20) == 0:
            print("Current patch prediction: " + str(key), "\r", end = "")

    return softmax_output.reshape(num_classes)


def prediction(softmax_output, renorm_nondiag = True, renorm_normal = False):
    """
    Implementation of inference algorithm for specimen-level probability distribution
    """
    
    # renormalize the elementwise softmax vector in order to return a valid probability distribution   
    renorm_dist = softmax_output/softmax_output.sum()

    if renorm_nondiag:
        renorm_dist[8] = 0 # set nondiagnostic class to zero
        renorm_dist/renorm_dist.sum() # renormalize the distribution

    if renorm_normal:
        if (renorm_dist[2] + renorm_dist[11] + renorm_dist[13]) > 0.9: # nonneoplastic classes
            return renorm_dist
        else:
            renorm_dist[[2, 11, 13]] = 0 # set nonneoplastic classes to zero
            return renorm_dist/renorm_dist.sum()
    
    # remove the nondiagnostic probability
    # renorm_dist = np.delete(renorm_dist, obj=8)

    return renorm_dist


def plot_srh_probability_histogram(renorm_dist, save_figure = False):
    """
    Plot (or save) figure with diagnosis and bar plot of probabilities for each mosaic.
    """
    # sort based on the renorm dist
    sorted_classes = [name for name, _ in sorted(zip(CLASS_NAMES, renorm_dist), key=lambda pair: pair[1], reverse = True)]

    plt.rcParams["figure.figsize"] = (10,10)
    plt.bar(x = sorted_classes, height = sorted(renorm_dist, reverse=True))
    plt.xticks(rotation = 90)
    plt.title(str(CLASS_NAMES[np.argmax(renorm_dist)]) + " (probability = " + str(np.round(np.max(renorm_dist), decimals=3)) + ")", fontsize=24)
    plt.subplots_adjust(bottom = 0.25)
    plt.show()

    if save_figure:
	    plt.savefig('gui_image.png', dpi = 500)
	    print("Figure saved to working directory.")


def plot_probablity_histogram(renorm_dist, save_figure = False):
    
    # generate a list of str indices 
    num_classes = [str(x) for x in list(range(len(renorm_dist)))] 
    # sort based on the renorm dist
    sorted_classes = [name for name, _ in sorted(zip(num_classes, renorm_dist), key=lambda pair: pair[1], reverse = True)] 

    plt.rcParams["figure.figsize"] = (10,10)
    plt.bar(x = sorted_classes, height = sorted(renorm_dist, reverse=True))
    plt.xticks(rotation = 90)
    plt.title("Class " + str(CLASS_NAMES[np.argmax(renorm_dist)]) + " (probability = " + str(np.round(np.max(renorm_dist), decimals=3)) + ")", fontsize=24)
    plt.subplots_adjust(bottom = 0.25)
    plt.show()

    if save_figure:
	    plt.savefig('gui_image.png', dpi = 500)
	    print("Figure saved to working directory.")


def directory_iterator(root):
    """
    Iterator through a directory that contains:
        1) individual subdirectories that contain SRH strips in alterating order
        2) Bijection of specimens to directories
    """

    pred_dict = defaultdict(list)
    for dirpath, dirname, files in os.walk(root): 
        if "NIO" in dirpath:
            try: 
	            mosaic = import_srh_dicom(dirpath)
	            patches = patch_generator(mosaic)
	            normalized_dist = prediction(feedforward(patches, model))
	            
	            pred_dict["specimen"].append(dirpath.split("/")[-1]) # select only the filename, remove root
	            
	            # append the predictions to each class key in the prediction dictionary
	            for i in range(len(CLASS_NAMES)):
	            	pred_dict[CLASS_NAMES[i]].append(normalized_dist[i])

            except:
	            continue

    return pred_dict


if __name__ == '__main__':
	   
    # load model    
    model = load_model("")
    
    # iterate through the directories with 
    root_dir = "" # root directory with each specimen in own directory 
    pred_dict = directory_iterator(root=root_dir)

    # save to results to spreadsheet
    df = DataFrame(pred_dict)
    df.to_excel("")
