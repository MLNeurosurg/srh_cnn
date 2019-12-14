
'''
Script to facilitate finding image patches that are diagnostic for a specific tumor class
'''

import os
import shutil
import numpy as np
import sys
from collections import defaultdict

from skimage.io import imread, imsave
import numpy as np
import shutil
import os
import sys
from keras.models import load_model

tumor_dict = {'ependymoma': 0,
 'glioblastoma': 1,
 'greymatter': 2,
 'lowgradeglioma': 3,
 'lymphoma': 4,
 'medulloblastoma': 5,
 'meningioma': 6,
 'metastasis': 7,
 'nondiagnostic': 8,
 'pilocyticastrocytoma': 9,
 'pituitaryadenoma': 10,
 'pseudoprogression': 11,
 'schwannoma':12,
 'whitematter': 13}

##### Variables to change
tumor_type = "whitematter"
n_images_per_case = 100
source_dir = os.path.join("/home/todd/Desktop/CNN_Images/columbia_trial_tiles", tumor_type)
dest_dir = os.path.join("/home/todd/Desktop/tsne_patches/columbia", tumor_type)


#####
incorrect_list = ["NIO10035", "NIO10036", "NIO10100", "NIO10101", "NIO10116", "NIO123", "NIO155", "NIO157", 
"NIO209", "NIO212", "NIO228", "NIO253", "NIO315", "NIO317", "NIO320"]

def nio_preprocessing_function(image):
	image[:,:,0] -= 102.1
	image[:,:,1] -= 91.0
	image[:,:,2] -= 101.5
	return(image)

def random_image_selector(patientlist, filelist):

	# output a dictionary with each patient as key and values is list of filenames

	patient_file_dict = defaultdict(list)
	for patient in patientlist:

		counter = 0
		patient_files = [filename for filename in filelist if patient in filename] # select only filename for a given patient

		selected_files = []

		while (len(selected_files) != n_images_per_case) and (counter < 1000):
			
			random_file = np.random.choice(patient_files)
			counter += 1

			if random_file in selected_files: # make sure not already selected
				continue

			# incorrect cases
			if patient in incorrect_list:
				selected_files.append(random_file) # just select random files

			else: 
				img = imread(source_dir + '/' + random_file)
				img = np.expand_dims(nio_preprocessing_function((img).astype(float)), axis=0)
				pred = model.predict(img, batch_size = 1, verbose = False)

				if pred[:,tumor_dict[tumor_type]] > 0.75: # ensure that representative image
					print(random_file + ": prob = " + str(pred[:,tumor_dict[tumor_type]]))
					selected_files.append(random_file)

			patient_file_dict[patient] = selected_files		

	return patient_file_dict


def selected_file_saver(image_dict):

	bad_dict = {}
	for patient, filelist in image_dict.items():
		if len(filelist) != n_images_per_case:
			bad_dict[patient] = len(filelist)

	for patient, filelist in image_dict.items():
		for file in filelist:
			shutil.copy(src = source_dir + "/" + file, dst = dest_dir + "/" + file)

	print(bad_dict)

if __name__ == '__main__':
	
	model = load_model("/home/todd/Desktop/Models/model_for_activations_Final_Resnet_weights.03-0.86.hdf5")

	filelist = os.listdir(source_dir) # list of all filenames
	filelist = [file for file in filelist if ("tif" in file) or ("png" in file)] # filter non image files
	patientlist = sorted(list(set([file.split("-")[0] for file in filelist]))) # sorted list of all patient names

	img_dict = random_image_selector(patientlist, filelist)
	selected_file_saver(img_dict)