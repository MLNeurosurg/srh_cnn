'''
Script to parse images based on CNN predictions for:
	1) Nondiagnostic
	2) Gliosis
	3) White matter
	4) Grey matter
'''

from skimage.io import imread, imsave
import numpy as np
import shutil
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121
from keras.utils import multi_gpu_model


from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D 
from keras.regularizers import l2 


img_rows = 300
img_cols = 300
total_classes = 14
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

parallel_model.load_weights("/home/todd/Desktop/Final_Resnet_weights.03-0.86.hdf5")

# directory containing images to predict on
source_dir = "/home/todd/Desktop/CNN_Images/nio_validation_tiles/meningioma"

# directories to move images
nondiag_dir = "/home/todd/Desktop/filtering_directory/nondiagnostic"
grey_dir = "/home/todd/Desktop/filtering_directory/greymatter"
white_dir = "/home/todd/Desktop/filtering_directory/whitematter"
gliosis_dir = "/home/todd/Desktop/filtering_directory/pseudoprogression"

# source directory length
source_dir_len = len(os.listdir(source_dir))

# dest directory lengths
nondiag_dir_len = len(os.listdir(nondiag_dir))
grey_dir_len = len(os.listdir(grey_dir))
white_dir_len = len(os.listdir(white_dir))
gliosis_dir_len = len(os.listdir(gliosis_dir))

def nio_preprocessing_function(image):
	image[:,:,0] -= 102.1
	image[:,:,1] -= 91.0
	image[:,:,2] -= 101.5
	return(image)

def file_tiler_saver(source_dir):

	counter = 0
	for dirs, root, files in os.walk(source_dir):

		files = sorted(files)

		for file in files:
			if "tif" in file or "png" in file:
				filename = file[0:(len(file)-4)]
				os.chdir(source_dir)
				img = imread(dirs + '/' + file)
				img = np.expand_dims(nio_preprocessing_function((img).astype(float)), axis=0)
				pred = model.predict(img, batch_size = 1, verbose = False)

				counter += 1   
				if np.argmax(pred) == 8:
				# if pred[:,8] > 0.5: # NONDIAGNOSTIC IMAGE
					shutil.move(src = source_dir + "/" + file, dst = nondiag_dir + "/" + file)
					print(filename + " > nondiagnostic")

				if np.argmax(pred) == 2:
				# if pred[:,3] > 0.5: # greymatter
					shutil.move(src = source_dir + "/" + file, dst = grey_dir + "/" + file)
					print(filename + " > greymatter")					

				if np.argmax(pred) == 13:
				# if pred[:,12] > 0.5: # whitematter
					shutil.move(src = source_dir + "/" + file, dst = white_dir + "/" + file)
					print(filename + " > whitematter")

				if np.argmax(pred) == 11:
				# if pred[:,11] > 0.5: # gliosis
					shutil.move(src = source_dir + "/" + file, dst = gliosis_dir + "/" + file)
					print(filename + " > gliosis")

				if counter % 1000 == 0:
					print(str(counter) + " of " + str(len(files))) 

				else: # DIAGNOSTIC IMAGE
					continue


def compare_dir_size(source_dir):
	
	source_post_move = len(os.listdir(source_dir))

	nondiag_dir_move = len(os.listdir(nondiag_dir))
	grey_dir_move = len(os.listdir(grey_dir))
	white_dir_move = len(os.listdir(white_dir))
	gliosis_dir_move = len(os.listdir(gliosis_dir))

	print("Source directory diff: " + str(source_post_move - source_dir_len))

	print("Nondiag. directory diff: " + str(nondiag_dir_move - nondiag_dir_len))
	print("Grey directory diff: " + str(grey_dir_move - grey_dir_len))
	print("White directory diff: " + str(white_dir_move - white_dir_len))
	print("Gliosis directory diff: " + str(gliosis_dir_move - gliosis_dir_len))
	print("Total move: " + str(nondiag_dir_move + grey_dir_move +  white_dir_move + gliosis_dir_move))

	if (nondiag_dir_move + grey_dir_move +  white_dir_move + gliosis_dir_move) == -(source_post_move - source_dir_len):
		print("Successful move!")

	else:
		print("Something went wrong! Please review directories.")


if __name__ == '__main__':

	file_tiler_saver(source_dir = source_dir)
	compare_dir_size(source_dir = source_dir)
