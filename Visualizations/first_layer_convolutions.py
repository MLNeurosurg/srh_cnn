#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:17:35 2018
@author: todd

# Evaluate first layer of convolutions for imagenet trained CNN
"""

from imageio import imread
from skimage.transform import resize
from scipy.misc import imresize
from keras.models import load_model
from keras.applications import InceptionResNetV2

def rescale(image):
	image -= image.mean()
	image /= image.std()
	image *= 64
	image += 128
	return image.clip(min=0, max=255).astype("uint8")

def plot_first_layer(model, interpolate = True):
    # Evaluate first layer of convolutions for trained CNN
    cnn_first_layer = model.get_layer(index = 1).get_weights()
    cnn_first_layer = np.asanyarray(cnn_first_layer)[0, : , :, :]

    for i in range(32):
        plt.subplot(4,8,i + 1)
        plt.axis("off")
        if interpolate:
            plt.imshow(imresize(rescale(cnn_first_layer[:,:,:,i]), (25, 25), interp="bicubic"))
        else:
            plt.imshow(rescale(cnn_first_layer[:,:,:,i]))
    plt.show()

# Load models of interest
model = load_model("/home/todd/Desktop/transfertrain_model.hdf5")
model_imagenet = InceptionResNetV2(weights="imagenet")

plot_first_layer(model, interpolate=False)
