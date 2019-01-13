'''
Script for testing tiling

'''

from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.misc import imresize
from numpy import resize
import numpy as np
import os
from skimage.exposure import equalize_adapthist, equalize_hist, adjust_gamma
from scipy.ndimage import median_filter, gaussian_filter, minimum_filter
from cv2 import imwrite
from keras.models import load_model


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

def channel_percentile_clip(matrix, percentile = 95):
	lower, upper = np.percentile(matrix, (100-percentile, percentile))
	clipped_matrix = matrix.clip(min = lower, max = upper)
	return clipped_matrix

# TESTED on NIO images, will need to test on INV images
def detect_dark_pixels(img):
	DNA, CH2, CH3 = return_channels(img)
	CH3 = channel_percentile_clip(CH3)
	CH2 = channel_percentile_clip(CH2)
	return (len(CH3[CH3 < 500]) + len(CH2[CH2 < 500])) ###150 count cutoff CH3 + CH2 

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
						# Value may need to titrated for GLIOMAS and less dense tissue
						if detect_dark_pixels(img_tile) >= 75000: # this was previously 60000
							
							os.chdir(nondiagnostic_dir)
							print(filename + "_" + str(img_number))
							imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						
						elif detect_dark_pixels(img_tile) < 75000:
							
							os.chdir(tiled_images_dir)
							print(filename + "_" + str(img_number))
							imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))

					else: # INV block
						if detect_dark_pixels(img_tile) < 175000:
							
							os.chdir(tiled_images_dir)
							print(filename + "_" + str(img_number))
							imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))

						# else detect_dark_pixels(img_tile) >= 175000:
						else:
							continue

							# os.chdir(nondiagnostic_dir)
							# print(filename + "_" + str(img_number))
							# imsave(filename + "_" + str(img_number) + ".tif", srh_preprocessing(img_tile))
						



if __name__ == '__main__':

	nondiagnostic_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/nondiagnostic"

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/ependymoma"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/ependymoma"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/glioblastoma"
	tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/glioblastoma"
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

	images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/meningioma"
	tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/meningioma"
	file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/metastasis"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/metastasis"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/nondiagnostic"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/nondiagnostic"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/pilocyticastrocytoma"
	tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/pilocyticastrocytoma"
	file_tiler_saver(images_to_read, tiled_images_dir)

	images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_NOTtiled/pituitaryadenoma"
	tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/pituitaryadenoma"
	file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/pseudoprogression"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/pseudoprogression"
	# file_tiler_saver(images_to_read, tiled_images_dir)

	# images_to_read = "/home/orringer-lab/Desktop/Training_Images/TIF_training_images/whitematter"
	# tiled_images_dir = "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training/whitematter"
	# file_tiler_saver(images_to_read, tiled_images_dir)



def main():
    """
    Function will MOVE DIAGNOSTIC IMAGES to destination folder. Will LEAVE NONdiagnostic images in working directory.
    
    """
    nondiagnostic_files = 0
    diagnostic_files = 0
    
    for file in filelist:
        image = img_to_array(load_img(file, target_size = (224, 224)))
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image, batch_size = 1, verbose = False)

        if pred[:,1] > 0.5: # NONDIAGNOSTIC IMAGE     
            nondiagnostic_files += 1
            print(file + ": " + str(pred) + " > " + "NON")
            
        else: # DIAGNOSTIC IMAGE
            shutil.move(src = scan_dir + "/" + file, dst = DIAGNOSTIC_dir + "/" + file)
            diagnostic_files += 1 
            print(file + ": " + str(pred) + " > " + "DIAGNOSTIC")

    print("Total files: " + str(len(filelist)))
    print("Total nondiagnostic: " + str(nondiagnostic_files))
    print("Total diagnostic: " + str(diagnostic_files))
