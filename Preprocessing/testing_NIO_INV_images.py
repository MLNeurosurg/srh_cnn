import shutil 
import os
from skimage.io import imread
import random

'''
Change Laser parameters at 053
See below
'''

NIO_newlaser = ["NIO053","NIO054","NIO055","NIO056","NIO057","NIO058","NIO059","NIO061","NIO062"]
num_files = 400

def sample_images(filelist, number_samples):
	NIO_sample = random.sample(filelist, number_samples)

	NIO_dict = {}
	for nio in NIO_sample:
		os.chdir(nio[0])
		img = imread(nio[2])
		NIO_dict[nio[2]] = img
	return NIO_dict

def mean_channel_function(img):
	# red = val[:,:,0].mean()
	CH2 = img[:,:,1].mean()
	CH3 = img[:,:,2].mean()
	return CH2, CH3

def sd_channel_function(img):
	CH2 = img[:,:,1].std()
	CH3 = img[:,:,2].std()
	return CH2, CH3

def import_function(directory):
	fileslist = []
	# for root, dirs, files in os.walk("/home/orringer-lab/Desktop/NIO_TIF_Files"):
	for root, dirs, files in os.walk(directory):	
		for file in files:
			if file.endswith(".tif"):
				fileslist.append((root, dirs, file))
	return fileslist

def import_newlaserfunction(directory):
	fileslist = []
	# for root, dirs, files in os.walk("/home/orringer-lab/Desktop/NIO_TIF_Files"):
	for root, dirs, files in os.walk(directory):	
		for file in files:
			for nioname in NIO_newlaser:
				if file.endswith(".tif") and (nioname in file):
					fileslist.append((root, dirs, file))
	return fileslist

def import_oldlaserfunction(directory):
	fileslist = []
	# for root, dirs, files in os.walk("/home/orringer-lab/Desktop/NIO_TIF_Files"):
	for root, dirs, files in os.walk(directory):	
		for file in files:
			for nioname in NIO_newlaser:
				if file.endswith(".tif") and (nioname not in file):
					fileslist.append((root, dirs, file))
	return fileslist

def stats_function(nio_dict, num_files = 400):
	NIO_CH2_mean = 0
	NIO_CH3_mean = 0
	NIO_CH2_std = 0
	NIO_CH3_std = 0
	for key, img in nio_dict.items():
		temp_CH2, temp_CH3 = mean_channel_function(img)
		temp_CH2_std, temp_CH3_std = sd_channel_function(img)
		NIO_CH2_mean += temp_CH2
		NIO_CH3_mean += temp_CH3
		NIO_CH2_std += temp_CH2_std
		NIO_CH3_std += temp_CH3_std
	
	NIO_CH2_mean_final = NIO_CH2_mean/num_files
	NIO_CH3_mean_final = NIO_CH3_mean/num_files
	NIO_CH2_std_final = NIO_CH2_std/num_files
	NIO_CH3_std_final = NIO_CH3_std/num_files

	return (NIO_CH2_mean_final, NIO_CH2_std_final), (NIO_CH3_mean_final, NIO_CH3_std_final)

if __name__ == '__main__':	


	NIO_filesold = import_oldlaserfunction("/home/orringer-lab/Desktop/NIO_TIF_Files")
	NIO_filesnew = import_newlaserfunction("/home/orringer-lab/Desktop/NIO_TIF_Files")
	INV_files = import_function("/media/orringer-lab/Seagate Expansion Drive/esteban_folder")
	
	NIO_dict_old = sample_images(NIO_filesold, num_files)	
	NIO_dict_new = sample_images(NIO_filesnew, num_files)
	INV_dict = sample_images(INV_files, num_files)


	CH2_old, CH3_old = stats_function(NIO_dict_old)
	print("NIOold_channel_CH2 mean = " + str(CH2_old[0]) + " and " + "std = " + str(CH2_old[1]))
	print("NIOold_channel_CH3 mean = " + str(CH3_old[0]) + " and " + "std = " + str(CH3_old[1]))


	CH2_new, CH3_new = stats_function(NIO_dict_new)
	print("NIOnew_channel_CH2 mean = " + str(CH2_new[0]) + " and " + "std = " + str(CH2_new[1]))
	print("NIOnew_channel_CH3 mean = " + str(CH3_new[0]) + " and " + "std = " + str(CH3_new[1]))


	CH2_inv, CH3_inv = stats_function(INV_dict)
	print("INV_channel_CH2 mean = " + str(CH2_inv[0]) + " and " + "std = " + str(CH2_inv[1]))
	print("INV_channel_CH3 mean = " + str(CH3_inv[0]) + " and " + "std = " + str(CH3_inv[1]))


	print("CH2old correction factor: " + str(CH2_inv[0]/CH2_old[0]))
	print("CH3old correction factor: " + str(CH3_inv[0]/CH3_old[0]))

	print("CH2new correction factor: " + str(CH2_inv[0]/CH2_new[0]))
	print("CH3new correction factor: " + str(CH3_inv[0]/CH3_new[0]))





'''
From NIO-053
orringer-lab@orringer-lab:~/Desktop/NIO_SRH$ python Testing_NIO_INV_images.py 
NIO_channel_CH2 mean = 708.83822052 and NIO_channel_CH2 std = 417.26130422
NIO_channel_CH3 mean = 1011.059358 and NIO_channel_CH3 std = 406.131540843

From the rest of the NIOs
NIO_channel_CH2 mean = 504.775945988 and NIO_channel_CH2 std = 268.676570689
NIO_channel_CH3 mean = 1798.46502916 and NIO_channel_CH3 std = 623.024677112


'''