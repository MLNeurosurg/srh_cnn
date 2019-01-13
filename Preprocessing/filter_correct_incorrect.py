


from skimage.io import imread, imsave
import numpy as np
import shutil
import os
import sys
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

tumor_type = "medulloblastoma"

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

img_rows = 300
img_cols = 300
total_classes = 14

# base_model = InceptionResNetV2(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
# # Add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(.5)(x)  
# x = Dense(20, kernel_initializer='he_normal')(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
# x = Dense(total_classes, kernel_initializer='he_normal')(x)
# predictions = Activation('softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)

# parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model.load_weights("/home/todd/Desktop/Models/Final_Resnet_weights.03-0.86PAPERMODEL.hdf5")

model = load_model("/home/todd/Desktop/Models/model_for_activations_Final_Resnet_weights.03-0.86.hdf5")

# directory containing images to predict on
source_dir = "/home/todd/Desktop/tsne_patches/" + tumor_type
# directories to move images
incorrect_dir = "/home/todd/Desktop/tsne_patches/incorrect_files"

# source directory length
source_dir_len = len(os.listdir(source_dir))
# dest directory lengths
incorrect_dir_length = len(os.listdir(incorrect_dir))

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
				# if np.argmax(pred) != index_to_filter:
				if pred[:,tumor_dict[tumor_type]] < 0.5:
					shutil.move(src = source_dir + "/" + file, dst = incorrect_dir + "/" + file)
					print(filename + " > incorrect")

				else: # DIAGNOSTIC IMAGE
					continue


def compare_dir_size(source_dir):
	source_post_move = len(os.listdir(source_dir))

	incorrect_dir_move = len(os.listdir(incorrect_dir))

	print("Source directory diff: " + str(source_post_move - source_dir_len))
	print("Nondiag. directory diff: " + str(incorrect_dir_move - incorrect_dir_length))


if __name__ == '__main__':

	file_tiler_saver(source_dir = source_dir)
	compare_dir_size(source_dir = source_dir)
