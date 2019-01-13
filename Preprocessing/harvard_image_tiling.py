'''
Script for testing tiling

Using tthi

'''

from skimage.io import imread, imsave
# from skimage.transform import resize
# from scipy.misc import imresize
from numpy import resize
import numpy as np
import os
from skimage.exposure import equalize_adapthist, equalize_hist, adjust_gamma
from scipy.ndimage import median_filter, gaussian_filter, minimum_filter
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img

model = load_model("/home/todd/Desktop/inception_model_trained_86acctrain.hdf5")

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

# def channel_percentile_clip(matrix, percentile = 95):
# 	lower, upper = np.percentile(matrix, (100-percentile, percentile))
# 	clipped_matrix = matrix.clip(min = lower, max = upper)
# 	return clipped_matrix

def min_max_rescaling(array):
	p_low, p_high = np.percentile(array, (3, 97))
	array = array.clip(min = p_low, max = p_high)
	img = (array - p_low)/(p_high - p_low)
	return img

def equalize_filter(array):
	DNA, CH2, CH3 = return_channels(array.astype('float'))
	img = np.empty((array.shape[0], array.shape[1], 3), dtype='float')

	img[:,:,0] = min_max_rescaling(DNA)
	img[:,:,1] = min_max_rescaling(CH2)
	img[:,:,2] = min_max_rescaling(CH3)

	img *= 255

	return img.astype(np.uint8)

def srh_preprocessing(array):
	inv_img = np.empty((array.shape[0], array.shape[1], 3), dtype=float)

	inv_img[:, :, 1] = array[:, :, 1] * 0.5 # Green/CH2 Channel, subtracted according to the methods section in SRS paper
	inv_img[:, :, 2] = array[:, :, 2] # Blue/CH3 Channel

	red_channel = np.subtract(inv_img[:, :, 2], (inv_img[:, :, 1])) # Subtract two channels
	red_channel = red_channel.clip(min = 0) # Send negative numbers to 1
	inv_img[:, :, 0] = red_channel # Red channel
	
	# inv_img *= 10 # Brighten the image by an order of magnitude
	return equalize_filter(inv_img.astype(float))

def file_tiler_saver(images_to_read, tiled_images_dir):
	total = 0
	counter = 0
	for dirs, root, files in os.walk(images_to_read):
		for file in files:
			if "tif" in file or "png" in file:
				filename = file[0:(len(file)-4)]
				os.chdir(images_to_read)
				img = imread(dirs + '/' + file)
				tile_list = tiling_images(img)
				
				for img_number, img_tile in enumerate(tile_list):
					pred_image = np.expand_dims(nio_preprocessing_function((srh_preprocessing(img_tile)).astype(float)), axis=0)
					pred = model.predict(pred_image, batch_size = 1, verbose = False)

					if pred[:,8] > 0.5: # NONDIAGNOSTIC IMAGE     
						os.chdir(nondiagnostic_dir)
						print(filename + "_" + str(img_number))
						imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						total += 1
					
					else: # DIAGNOSTIC IMAGE
						os.chdir(tiled_images_dir)
						print(filename + "_" + str(img_number))
						imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))

					# if "NIO" in filename: # NIO block
					# 	# Value may need to titrated for GLIOMAS and less dense tissue
					# 	if detect_dark_pixels(img_tile) >= 75000: # this was previously 60000
							
					# 		os.chdir(nondiagnostic_dir)
					# 		print(filename + "_" + str(img_number))
					# 		imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						
					# 	elif detect_dark_pixels(img_tile) < 75000:
							
					# 		os.chdir(tiled_images_dir)
					# 		print(filename + "_" + str(img_number))
					# 		imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))

					# else: # INV block
					# 	if detect_dark_pixels(img_tile) < 175000:
							
					# 		os.chdir(tiled_images_dir)
					# 		print(filename + "_" + str(img_number))
					# 		imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))

					# 	# else detect_dark_pixels(img_tile) >= 175000:
					# 	else:
					# 		continue

					# 		# os.chdir(nondiagnostic_dir)
					# 		# print(filename + "_" + str(img_number))
					# 		# imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						


if __name__ == '__main__':

	nondiagnostic_dir = "/home/todd/Desktop/CNN_Images/Harvard_Images/harvard_trial_images/nondiagnostic"

	images_to_read = "/home/todd/Desktop/CNN_Images/Harvard_Images/harvard_tiled_images/glioblastoma"
	tiled_images_dir = "/home/todd/Desktop/CNN_Images/Harvard_Images/harvard_trial_images/glioblastoma"
	file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/lowgradeglioma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/tester/lowgradeglioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/columbia_NotTiled/greymatter"
	# tiled_images_dir = "/home/todd/Desktop/CNN_Images/columbia_trial_tiles/greymatter"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/lowgradeglioma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/lowgradeglioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/lymphoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/tester/lymphoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/medulloblastoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/medulloblastoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)
	
	# images_to_read = "/home/todd/Desktop/CNN_Images/columbia_NotTiled/meningioma"
	# tiled_images_dir = "/home/todd/Desktop/CNN_Images/columbia_trial_tiles/meningioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/metastasis"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/tester/metastasis"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/nondiagnostic"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/nondiagnostic"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/pilocyticastrocytoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/tester/pilocyticastrocytoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/todd/Desktop/CNN_Images/columbia_NotTiled/pituitaryadenoma"
	# tiled_images_dir = "/home/todd/Desktop/CNN_Images/columbia_trial_tiles/pituitaryadenoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/pseudoprogression"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/tester/pseudoprogression"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/whitematter"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/tester/whitematter"
	# file_tiler_saver(images_to_read, tiled_images_dir)


