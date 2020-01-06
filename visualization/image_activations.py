#!/usr/bin/env python3

from keras.models import load_model
from keras.models import Model
from imageio import imread
import numpy as np
import sys
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from keras.applications import InceptionResNetV2

from preprocessing import cnn_preprocessing

def rescale(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    return image.clip(min = 0, max=255).astype("uint8")

def plot_activations(model, image, activation_layer):
    activations_model = Model(inputs=model.input, outputs = model.layers[activation_layer].output) 
    activations = activations_model.predict(cnn_preprocessing(image[None,:,:,:]))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.axis("off")
        plt.imshow(rescale(activations[0,:,:,i]))
    plt.show()

def plot_activations_imagenet(model, image, activation_layer):
    activations_model = Model(inputs=model.input, outputs = model.layers[activation_layer].output) 
    activations = activations_model.predict(image[None,:,:,:] - image.mean())
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.axis("off")
        plt.imshow(rescale(activations[0,:,:,i]))
    plt.show()

if __name__ == "__main__":


    # load models
    model = load_model("")
    # model_imagenet = InceptionResNetV2(weights="imagenet")

    # import single image
    img = imread("").astype(np.float64)

    plot_activations(model, img, activation_layer = 3)

