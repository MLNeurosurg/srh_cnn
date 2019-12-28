
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


def feedforward(patch_array, model):
    """
    Function to perform a forward pass, with preprocessing on all the patches generated above, outputs 
    """
    num_patches = patch_array.shape[0]

    softmax_output = np.zeros((1,TOTAL_CLASSES), dtype=float)
    nondiag_count = 0
    
    for i in range(num_patches):
        patch = cnn_preprocessing(patch_array[i,:,:,:])
        pred = model.predict(patch[None,:,:,:], batch_size = 1)
        softmax_output += pred # unnormalized probabability distribution

    return softmax_output.reshape(TOTAL_CLASSES)


def prediction(softmax_output, renormalize = True):
    """
    Implementation of inference algorithm for specimen-level probability distribution
    """
        
    # renormalize the elementwise softmax vector in order to return a valid probability distribution   
    renorm_dist = softmax_output/softmax_output.sum()

    if renormalize:
        if (renorm_dist[2] + renorm_dist[11] + renorm_dist[13]) > 0.9: # nonneoplastic classes
            return renorm_dist
        else:
            renorm_dist[[2, 11, 13]] = 0 # set nonneoplastic classes to zero
            return renorm_dist/renorm_dist.sum()

    if not renormalize:
        return renorm_dist


def directory_iterator(root):
    """
    Iterator through a directory that contains:
        1) individual subdirectories that contain SRH strips in alterating order
        2) Bijection of specimens to directories
    """

    def remove_nondiag(norm_dist):
        norm_dist[0] = 0 # set nondiagnostic class to zero
        return norm_dist/norm_dist.sum() # renormalize the distribution

    pred_dict = defaultdict(list)
    for dirpath, dirname, files in os.walk(root): 
        if "NIO" in dirpath:
            try: 
	            mosaic = import_dicom(dirpath)
	            patches = patch_generator(mosaic)
	            normalized_dist = prediction(feedforward(patches, model))
	            
	            pred_dict["specimen"].append(dirpath.split("/")[-1]) # select only the filename, remove root
	            
	            # append the predictions to each class key in the prediction dictionary
	            for i in range(len(CLASS_NAMES)):
	            	pred_dict[class_names[i]].append(normalized_dist[i])

	        except:
	            continue

    return pred_dict


def plot_probability_histogram(renorm_dist, save_figure = False):
    """
    Plot (or save) figure with diagnosis and bar plot of probabilities for each mosaic.
    """
    sorted_classes = [name for name, _ in sorted(zip(CLASS_NAMES, renorm_dist), key=lambda pair: pair[1], reverse = True)]

    plt.rcParams["figure.figsize"] = (10,10)
    plt.bar(x = sorted_classes, height = sorted(renorm_dist, reverse=True))
    plt.xticks(rotation = 90)
    plt.title(str(class_dict[np.argmax(renorm_dist)]) + " (probability = " + str(np.round(np.max(renorm_dist), decimals=3)) + ")", fontsize=24)
    plt.subplots_adjust(bottom = 0.25)
    plt.show()

    if save_figure:
	    plt.savefig('gui_image.png', dpi = 500)
	    print("Figure saved to working directory.")


if __name__ == '__main__':
	   
	# constants
    IMAGE_SIZE, IMAGE_CHANNELS = 300, 3
    TOTAL_CLASSES = 14
	CLASS_NAMES = ['ependymoma',
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

    # load model    
    model = load_model("")
    
    # iterate through the directories with 
    root_dir = "" # root directory with each specimen in own directory 
    pred_dict = directory_iterator(root=root_dir)

    # save to results to excel spreadsheet
    df = DataFrame(pred_dict)
    df.to_excel("")
