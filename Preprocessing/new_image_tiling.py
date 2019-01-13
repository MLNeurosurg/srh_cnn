'''
Script for testing tiling

'''

from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.misc import imresize
from numpy import resize
import numpy as np
import os

def return_channels(array):
	return array[:,:,0], array[:,:,1], array[:,:,2]

def tiling_images(image):
	tile_list = []
	tile_size = 300
	starts = [0, 100, 200, 300, 400, 500, 600, 700]

	for y in starts:
		for x in starts:
			tile_list.append(image[y:y+tile_size, x:x+tile_size, :])
	return tile_list

# def channel_percentile_clip(matrix, percentile = 3):
# 	lower, upper = np.percentile(matrix, (100-percentile, percentile))
# 	clipped_matrix = matrix.clip(min = lower, max = upper)
# 	return clipped_matrix

def high_variance(matrix):
	DNA, CH2, CH3 = return_channels(matrix)
	lower, upper = np.percentile(CH3, (10, 90))
	CH3new = CH3[CH3 > lower] 
	CH3new = CH3[CH3 < upper]
	return np.std(CH3new)

def detect_dark_pixels(img):
	"""
	Use the CH3 images because the laser for these images has not changed significantly
	"""
	DNA, CH2, CH3 = return_channels(img)
	lower, upper = np.percentile(CH3, (10, 90))
	CH3new = CH3[CH3 > lower] 
	CH3new = CH3[CH3 < upper]
	return len(CH3new[CH3new < 1500])/(300**2) # 1500 is about the mean value of brightness for EMPTY SPACE

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
	if np.all(array[:,:,0] == 0):  # Test for black red channel
		# INV PATH
		inv_img = np.empty((array.shape[0], array.shape[1], 3), dtype=np.uint16)

		inv_img[:, :, 1] = array[:, :, 1] * 1.40  # Green/CH2 Channel 
		inv_img[:, :, 2] = array[:, :, 2] * 2.21 # Blue/CH3 Channel

		red_channel = np.subtract(inv_img[:, :, 2], inv_img[:, :, 1]) # Subtract two channels
		red_channel = red_channel.clip(min = 0) # Send negative numbers to 1
		inv_img[:, :, 0] = red_channel # Red channel
		
		# inv_img *= 10 # Brighten the image by an order of magnitude
		return equalize_filter(inv_img.astype(np.uint16))
	else:        
		return equalize_filter(array.astype(np.uint16))

def file_tiler_saver(images_to_read, tiled_images_dir):
	for dirs, root, files in os.walk(images_to_read):
		for file in files:
			if "tif" in file:
				filename = file[0:(len(file)-4)]
				os.chdir(images_to_read)
				img = imread(dirs + '/' + file)[0:1000, 0:1000]
				tile_list = tiling_images(img)
				
				for img_number, img_tile in enumerate(tile_list):
					
					if "NIO" in filename: # NIO block
						if (detect_dark_pixels(img_tile) < 0.50) and (high_variance(img_tile) > 200): 

							print("Diagnostic image: " + filename + "_" + str(img_number))
							print("Variance: " + str(high_variance(img_tile)))
							print("Dark Pixels: " + str(detect_dark_pixels(img_tile)))

							os.chdir(tiled_images_dir)
							imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						
						else:

							print("NONdiagnostic image: " + filename + "_" + str(img_number))
							print("Variance: " + str(high_variance(img_tile)))
							print("Dark Pixels: " + str(detect_dark_pixels(img_tile)))

							os.chdir(nondiagnostic_dir)
							imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						
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

	nondiagnostic_dir = "/home/orringer-lab/Desktop/test_images/nondiagnostic"

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/ependymoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/ependymoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/metastasis"
	tiled_images_dir = "/home/orringer-lab/Desktop/test_images/metastasis"
	file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/greymatter"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/greymatter"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/lowgradeglioma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/lowgradeglioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/lymphoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/lymphoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/medulloblastoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/medulloblastoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/meningioma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/meningioma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/metastasis"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/metastasis"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/nondiagnostic"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/nondiagnostic"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/pilocyticastrocytoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/pilocyticastrocytoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/pituitaryadenoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/pituitaryadenoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/pseudoprogression"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/pseudoprogression"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/whitematter"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/whitematter"
	# file_tiler_saver(images_to_read, tiled_images_dir)
