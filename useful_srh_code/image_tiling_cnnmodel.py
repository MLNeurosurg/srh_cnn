'''
Script for tiling

'''

from skimage.io import imread, imsave
from numpy import resize
import numpy as np
import os
# from skimage.exposure import equalize_adapthist, equalize_histq, adjust_gamma
# from scipy.ndimage import median_filter, gaussian_filter, minimum_filter
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121
from keras.utils import multi_gpu_model

from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D 

from preprocessing import return_channels, min_max_rescaling, channel_rescaling, cnn_preprocessing
from model_training import srh_model

img_rows = 300
img_cols = 300
total_classes = 14

def tiling_images(image):
	tile_list = []
	tile_size = 300
	starts = [0, 100, 200, 300, 400, 500, 600, 700]
	for y in starts:
		for x in starts:
			tile_list.append(image[y:y+tile_size, x:x+tile_size, :])
	return tile_list

def srh_preprocessing(array):
	if np.all(array[:,:,0] == 0):  # Test for black red channel
		# INV PATH
		inv_img = np.empty((array.shape[0], array.shape[1], 3), dtype=float)

		inv_img[:, :, 1] = array[:, :, 1] * 1.40  # Green/CH2 Channel 
		inv_img[:, :, 2] = array[:, :, 2] * 2.21 # Blue/CH3 Channel

		red_channel = np.subtract(inv_img[:, :, 2], inv_img[:, :, 1]) # Subtract two channels
		red_channel = red_channel.clip(min = 0) # Send negative numbers to 1
		inv_img[:, :, 0] = red_channel # Red channel
		
		# inv_img *= 10 # Brighten the image by an order of magnitude
		return channel_rescaling(inv_img.astype(np.uint16))
	else:      
		# NIO PATH  
		return channel_rescaling(array.astype(np.uint16))

def file_tiler_saver(images_to_read, tiled_images_dir):
	for dirs, root, files in os.walk(images_to_read):
		for file in files:
			if "tif" in file:
				filename = file[0:(len(file)-4)]
				os.chdir(images_to_read)
				img = imread(dirs + '/' + file)[0:1000, 0:1000]
				tile_list = tiling_images(img)
				
				for img_number, img_tile in enumerate(tile_list):
					pred_image = np.expand_dims(cnn_preprocessing((srh_preprocessing(img_tile)).astype(float)), axis=0)
					pred = model.predict(pred_image, batch_size = 1, verbose = False)

					if np.argmax(pred) == 8: # NONDIAGNOSTIC IMAGE     
						os.chdir(nondiagnostic_dir)
						print(filename + "_" + str(img_number))
						imsave(filename + "_" + str(img_number) + ".png", srh_preprocessing(img_tile))
					
					else: # DIAGNOSTIC IMAGE
						os.chdir(tiled_images_dir)
						print(filename + "_" + str(img_number))
						imsave(filename + "_" + str(img_number) + ".png", srh_preprocessing(img_tile))		


if __name__ == '__main__':


	parallel_model = srh_model()
 	parallel_model.load_weights("/home/todd/Desktop/Models/Final_Resnet_weights.03-0.86PAPERMODEL.hdf5")

	nondiagnostic_dir = "/home/todd/Desktop/NIO_088/nondiagnostic"

	images_to_read = "/home/todd/Desktop/NIO_088/tif_holder"
	tiled_images_dir = "/home/todd/Desktop/NIO_088/lymphoma"
	file_tiler_saver(images_to_read, tiled_images_dir)

