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

# NIO_numbers_test = {
# "ependymoma": [],
# "glioblastoma": [],
# "greymatter": [],
# "lowgradeglioma": [],
# "lymphoma": [],
# "medulloblastoma": [],
# "meningioma": [],
# "metastasis": [],
# "nondiagnostic": [],
# "pilocyticastrocytoma": [],
# "pituitaryadenoma": [],
# "pseudoprogression": [],
# "whitematter": []
# }

img_rows = 300
img_cols = 300
total_classes = 14
# base_model = DenseNet121(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
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

parallel_model = multi_gpu_model(model, gpus=2)

parallel_model.load_weights("/home/todd/Desktop/Models/Final_Resnet_weights.03-0.86PAPERMODEL.hdf5")

def return_channels(array):
	return array[:,:,0], array[:,:,1], array[:,:,2]

def nio_preprocessing_function(image):
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return(image)

def tiling_images(image):
	tile_list = []
	tile_size = 300
	starts = [0, 100, 200, 300, 400, 500, 600, 700]
	for y in starts:
		for x in starts:
			tile_list.append(image[y:y+tile_size, x:x+tile_size, :])
	return tile_list

def min_max_rescaling(array):
	p_low, p_high = np.percentile(array, (3, 97))
	array = array.clip(min = p_low, max = p_high)
	img = (array - p_low)/(p_high - p_low)
	return img

def channel_rescaling(array):
	DNA, CH2, CH3 = return_channels(array.astype('float'))
	img = np.empty((array.shape[0], array.shape[1], 3), dtype='float')

	img[:,:,0] = min_max_rescaling(DNA)
	img[:,:,1] = min_max_rescaling(CH2)
	img[:,:,2] = min_max_rescaling(CH3)

	img *= 255

	return img.astype(np.uint8)

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
					pred_image = np.expand_dims(nio_preprocessing_function((srh_preprocessing(img_tile)).astype(float)), axis=0)
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

	nondiagnostic_dir = "/home/todd/Desktop/NIO_088/nondiagnostic"

	images_to_read = "/home/todd/Desktop/NIO_088/tif_holder"
	tiled_images_dir = "/home/todd/Desktop/NIO_088/lymphoma"
	file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/tif_files/umich_NOTtiled/glioblastoma"
	# tiled_images_dir = "/home/todd/Desktop/filtering_directory/glioblastoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/tif_files/umich_NOTtiled/lowgradeglioma"
	# tiled_images_dir = "/home/todd/Desktop/filtering_directory/lowgradeglioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/tif_files/umich_NOTtiled/meningioma"
	# tiled_images_dir = "/home/todd/Desktop/filtering_directory/meningioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/tif_files/umich_NOTtiled/metastasis"
	# tiled_images_dir = "/home/todd/Desktop/filtering_directory/metastasis"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/tif_files/umich_NOTtiled/pituitaryadenoma"
	# tiled_images_dir = "/home/todd/Desktop/filtering_directory/pituitaryadenoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)



