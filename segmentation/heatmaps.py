#!/usr/bin/env python3

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

from keras.models import load_model  

from preprocessing.preprocess import return_channels, channel_rescaling, cnn_preprocessing 
from preprocessing.io import import_srh_dicom
from preprocessing.patch_generator import starts_finder

IMAGE_SIZE, IMAGE_CHANNELS = 300, 3
TOTAL_CLASSES = 3
STEP_SIZE = 100

# define patch object 
class Patch(object):
	def __init__(self, patch_num, num_classes):
		self.patch_num = patch_num
		self.tile_indices = 0
		self.classes = num_classes
		self.softmax = np.zeros((num_classes))

	def __str__(self):
		return("patch_" + str(self.patch_num))

	def __repr__(self):
		return("patch_" + str(self.patch_num))

	def set_indices(self, index):
		self.tile_indices = index

	def get_indices(self):
		return self.tile_indices

	def set_softmax(self, softmax):
		self.softmax = softmax

	def get_softmax(self):
		return(self.softmax)

def indices_map(array, step_size = STEP_SIZE): 
	"""
	This is an index map the same size as the input image that is used to identify which heatmap pixels overlap with each patch
	"""
	# define the number of values needed in your matrix if indices
	array_values = np.arange(array.shape[0]/step_size * array.shape[1]/step_size).astype(int)
	
	# initialize an empty matrix
	init_array = np.empty(shape=(array.shape[0], array.shape[1]))
	
	# generate starting points for for-loop
	starts = np.arange(array.shape[0], step = step_size) # this assumes a square matrix 
	counter = 0 # counter for indexing into array_values
	for y in starts:
		for x in starts:
			fill_array = np.zeros((step_size, step_size))
			fill_array.fill(array_values[counter]) # fill with index value
			init_array[y:y + step_size, x:x + step_size] = fill_array
			counter += 1 
	
	return init_array

def patch_dictionary(image, model, indices_map, image_preprocced = False, step_size = STEP_SIZE):
	"""
	Function that generates a set of patches as values and the patch_number is the key in the dictionary
	key = patch_number
	value = patch_object
	"""
	assert image.shape[0] == indices_map.shape[0], "Image and indices_map are different dimensions"
	assert step_size <= IMAGE_SIZE, "Step size is too large. Must be less than 300."

	starts = starts_finder(side = image.shape[0], stride = STEP_SIZE, image_size = IMAGE_SIZE)
	patch_dict = {}
	counter = 0
	for y in starts:
		for x in starts:
			patch_dict[counter] = Patch(patch_num = counter, num_classes = TOTAL_CLASSES)

			# preprocess patch
			patch = image[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE, :]
			
			if not image_preprocced:
				patch = channel_rescaling(patch)

			# forward pass
			patch = cnn_preprocessing(patch)
			pred = model.predict(patch[None,:,:,:], batch_size = 1)

			# update patch_object
			patch_dict[counter].set_softmax(pred)
			patch_dict[counter].set_indices(np.unique(indices_map[y:y + IMAGE_SIZE, x:x + IMAGE_SIZE]))
			counter += 1
			print(counter)

	return patch_dict


def srh_heatmap(patch_dict, image_size, step_size = STEP_SIZE):
	"""
	Function that takes a patch dictionary and mosaic_size (eg mosaic.shape[0])
	and return a k-channel heatmap, where k is the number of output classes
	"""
	# define the number of pixels in the heatmap
	heatmap_pixels = int(np.square(image_size/step_size))
	
	# initialize the heatmap dictionary
	heatmap_dict = {}
	for pixel in range(heatmap_pixels):
		heatmap_dict[pixel] = np.zeros((TOTAL_CLASSES)) 

	# this is the patch-pixel matching step that does not scale well (n^2), painfully slow
	for pixel in range(heatmap_pixels):
		print(pixel)
		for patch, patch_object in patch_dict.items():
			if pixel in patch_object.get_indices():
				heatmap_dict[pixel] += patch_object.get_softmax().reshape((TOTAL_CLASSES))

	# convert the dictionary to an array
	flattened_image = np.zeros((heatmap_pixels, TOTAL_CLASSES))
	for pixel, unormed_dist in heatmap_dict.items():
		flattened_image[pixel, :] = unormed_dist/unormed_dist.sum() # renormalize the distribution
		
	# convert array to K-channel image, where k is the number of classes
	height_width = int(np.sqrt(heatmap_pixels))
	heatmap = flattened_image.reshape((height_width, height_width, TOTAL_CLASSES))
	return heatmap

def display_image(heatmap, class_index):
	"""
	Function to display heatmap results for a 
	"""
	plt.imshow(heatmap[:,:,class_index])
	plt.show()

def save_heatmap_image(file_name, heatmap, class_index, cmap='Greys_r'):
	"""
	Saves the greyscale heatmap image for specific class
	"""
	plt.imsave(file_name, heatmap[:,:,class_index], cmap='Greys_r', vmin=0,vmax=1)


if __name__ == '__main__':

    # import model and specify image directory
    model = load_model("/home/todd/Desktop/RecPseudo_project/patches/cv_round2/recurmodel_kthfold_2.hdf5")
    image_dir = "/home/todd/Desktop/Recurrence_mosaics_figures/NIODS10070_2"

    # import image
    mosaic = import_srh_dicom(image_dir)
    image_size = mosaic.shape[0]

    # index map to allow for patch-tile mapping
    heatmap_indices = indices_map(mosaic, step_size = STEP_SIZE)

    # predict on each patch
    patch_dict = patch_dictionary(mosaic, model, heatmap_indices, image_preprocced=False, step_size = STEP_SIZE)

    # generate heatmap predictions
    heatmap = srh_heatmap(patch_dict, image_size, step_size = STEP_SIZE)

	plt.imshow(heatmap[:,:,1])
	plt.show()