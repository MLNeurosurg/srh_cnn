#!/usr/bin/env python3

'''
Script to build our SRH model
'''
# Model Layers
from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D
from keras.utils import multi_gpu_model

# Open-source models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121

TOTAL_CLASSES = 14

def srh_model(backbone, input_shape = (300, 300, 3), weights = False, dropout = 0.5, gpu_num = 1):
	"""
	SRH CNN model import
	"""
	if weights: 
		base_model = backbone(weights="imagenet", include_top=False, input_shape = input_shape)

	else: 
		base_model = backbone(weights=None, include_top=False, input_shape = input_shape)

	x = base_model.output
	x = GlobalAveragePooling2D(name = "srh_global_average_pool")(x) # Add a global spatial average pooling layer
	x = Dropout(dropout, name = "srh_dropout")(x)
	x = Dense(20, kernel_initializer='he_normal', name = "srh_dense")(x)
	x = BatchNormalization(name='srh_batch_norm')(x)
	x = Activation("relu", name='srh_activation')(x)
	x = Dense(TOTAL_CLASSES, kernel_initializer='he_normal', name = "srh_dense_2")(x)
	predictions = Activation('softmax', name='srh_activation_2')(x)
	model = Model(inputs=base_model.input, outputs=predictions)

	# Distribute model across GPUs
	if gpu_num > 1: 
		parallel_model = multi_gpu_model(model, gpus=gpu_num)
		return parallel_model

	else:
		return model

if __name__ == "__main__":

	model = srh_model(InceptionResNetV2, gpu_num = 1)