# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import seaborn as sns
from keras import models
from skimage.io import imread, show
from skimage.transform import resize
from skimage import filters
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D 

from collections import defaultdict
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import backend as K
import cv2

# /home/todd/Desktop/Models/Final_Resnet_weights.05-0.86.hdf5
img_rows = 300
img_cols = 300
img_channels = 3
total_classes = 14

model = load_model("/home/todd/Desktop/transfertrain_model.hdf5")
#model = InceptionResNetV2(weights="imagenet")

##### Visualizing Convnet filters using activation maximization
layer_outputs_all = [layer.output for layer in model.layers]
layer_outputs = []
layer_names = []
for layer in layer_outputs_all:
    if "conv" in layer.name: # Capture only the activation layers
        layer_names.append(layer.name)
        layer_outputs.append(layer)
layer_names

#Define regularizations:
def blur_regularization(img, grads, size = (5,5)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

def process_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    return x.astype(np.float64)

def rescale_image(img):
    return (img - img.min())/(img.max() - img.min())

def gradient_ascent_iteration(loss_function, img):
    loss_value, grads_value = loss_function([img])    
    gradient_ascent_step = img + grads_value * 0.9

    #Convert to row major format for using opencv routines
#    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
#    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))
    
    grads_row_major = grads_value[0,:,:,:] 
    img_row_major = gradient_ascent_step[0,:,:,:]

    #List of regularization functions to use
    regularizations = [blur_regularization, decay_regularization, clip_weak_pixel_regularization]

    #The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    #Convert image back to 1 x height x width X 3 
#    img = np.float32([np.transpose(img, (0, 1, 2))])
    img = img[None, :,:,:].astype(np.float32)
    return img

def generate_pattern(layer_name, filter_index, size=150):
    '''
    Function that uses gradient ascent to maximize the activation of a convolutional layerto help with visualization
    '''
    layer_output = model.get_layer(layer_name).output
    
    loss = K.mean(layer_output[:,:,:,filter_index]) #loss function: mean value of input image 
    grads = K.gradients(loss, model.input)[0] #gradients with respect to loss function and given input image
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #normalize the gradient tensor by dividing by its L2 norm
    iterate = K.function([model.input], [loss, grads]) 

    input_img_data = np.random.random((1, size, size, 3)) # initalize random noise for input image
    input_img_data -= input_img_data.mean() * 25 # mean center and multiple by variance (around 50 for most images)

    for i in range(100): # runs gradient ASCENT for 100 steps
        input_img_data = gradient_ascent_iteration(iterate, input_img_data)

    img = input_img_data[0]
    return process_image(img)
test = generate_pattern("conv2d_159", 12)
plt.imshow(test)

# Generating a grid of all filter response patterns in a layer
def filter_activation_grid(layer_name, size = 64, margin = 5):
    
    side = int(np.sqrt(size))
    filter_results = {}
    results = np.zeros((side * size + (side-1) * margin, side * size + (side-1) * margin, 3), dtype=np.float64)
    for i in range(side):
        for j in range(side):
            print(i, j)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            
            try:
                filter_img = generate_pattern(layer_name, i + (j * side), size=size)
#                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filters.gaussian(filter_img, multichannel=True, preserve_range=True)
                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = filter_img
                filter_results[layer_name + "_" + str(i + (j * side))] = filter_img
                
            except: # if layers has less than the number of filters, will fill those with black images
                results[horizontal_start:horizontal_end, vertical_start:vertical_end, :] = np.zeros((size, size, 3))
            
    plt.figure(figsize=(20,20)) 
    plt.imshow(results) # show the results
    
    return filter_results
filter_dict = filter_activation_grid("conv2d_3")


def generate_pattern_time_series(layer_name, filter_index, iterations_record, size=150):
    '''
    Function that uses gradient ascent to maximize the activation of a convolutional layerto help with visualization
    '''
    layer_output = model.get_layer(layer_name).output
    
    loss = K.mean(layer_output[:,:,:,filter_index]) #loss function: mean value of input image 
    grads = K.gradients(loss, model.input)[0] #gradients with respect to loss function and given input image
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) #normalize the gradient tensor by dividing by its L2 norm
    iterate = K.function([model.input], [loss, grads]) 

    input_img_data = np.random.random((1, size, size, 3)) # initalize random noise for input image
    input_img_data -= input_img_data.mean() * 25 # mean center and multiple by variance (around 50 for most images)

    num_images = len(iterations_record) + 1
    margin = 5
    results = np.zeros((150, (size * num_images) + (num_images - 1) * margin, 3), dtype=np.float64)
    
    iterations = iterations_record[-1]
    results[0:150, 0:150, :] = process_image(((np.random.random((1, size, size, 3)) * 1)[0])) # sd 25
    location = 0
    
    for i in range(iterations + 1):
        input_img_data = gradient_ascent_iteration(iterate, input_img_data)
        if i in iterations_record:
            print(i)
            location += 150 + margin
            results[:, location:location + 150, :] = process_image(input_img_data[0])
            
    return results
iterations = (1, 5, 10, 20, 50, 100, 200, 500)
img = generate_pattern_time_series(layer_name = "conv2d_159", filter_index = 70, iterations_record = iterations) # DEFINITELY USE 8, 14 IS BEST
plt.imshow(img)



#################### Visualizing activations 
layer_outputs_all = [layer.output for layer in model.layers]
layer_outputs = []
layer_names = []
for layer in layer_outputs_all:
    if "act" in layer.name: # Capture only the activation layers
        layer_names.append(layer.name)
        layer_outputs.append(layer)
layer_names

activation_model = models.Model(input=model.inputs, outputs=model.get_layer("activation_159").output) # indexing into the global average pooling layer 

def nio_preprocessing_function(image):
    """
    Channel-wise means calculated over INV dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

def find_max_activation(activation):
    
    bright_img = np.zeros((activation.shape[0], activation.shape[1]))
    
    for i in range(activation.shape[2]):
        img_mean = activation[:,:,i].mean()
        
        if img_mean > bright_img.mean():
            bright_img = activation[:,:,i]
            
    return bright_img
test = find_max_activation(activation)

def import_images(directory):
    filelist = os.listdir(directory)
    imagelist = []
    for file in filelist:
        imagelist.append(imread(directory + "/" + file))
    return imagelist
image_list = import_images("/home/todd/Desktop/activation_max_figures/Images_for_activation_maps/NIO105_MET")
    

def activation_statistics(img_list, filter_num):
    act_list = []
    for i in image_list:
        activation = activation_model.predict(nio_preprocessing_function(i.astype(np.float64))[None,:,:,:]) # predict on each image
        activation = activation[0,:,:,filter_num].mean() # index into the filter of interest
        act_list.append(activation)
    
    return act_list

def layer_statistics(img_list):
    
    def activations(img_list):
        
        act_list = []
        for i in image_list:
            activation = activation_model.predict(nio_preprocessing_function(i.astype(np.float64))[None,:,:,:]) # predict on each image
            act_list.append(activation)
        return act_list
    
    filter_dict = defaultdict(list)
    activations_list = activations(img_list)
    for act_map in activations_list:
        for layer_filter in range(activation_model.output_shape[3]):
            filter_dict[layer_filter].append(np.mean(act_map[0,:,:,layer_filter]))
    
    filter_average = {}
    for layer_filter, means in filter_dict.items():
        filter_average[layer_filter] = np.mean(means)
        
    return filter_average

#filter_dict = layer_statistics(image_list)
#sorted(filter_dict, key=filter_dict.get, reverse=True)


filter_num = 148
img = activation_statistics(image_list, filter_num)
sns.distplot(img, bins=100)
plt.xlim(0, 2)


dist_dict = {}
dist_dict["gm_159"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["gm_12"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["gm_148"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["gbm_159"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["gbm_12"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["gbm_148"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["met_159"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["met_12"] = np.random.choice(img, size = 1000, replace = False)
dist_dict["met_148"] = np.random.choice(img, size = 1000, replace = False)


import pandas as pd
df = pd.DataFrame((dist_dict))
df.to_excel("dist_activ.xlsx")
img = imread("/home/todd/Desktop/activation_max_figures/NIO101-3728_2_tile_005_59.tif")
plt.imshow(img)
filter_num = 148
plt.imshow(activation_model.predict(nio_preprocessing_function(img.astype(np.float64))[None,:,:,:])[0,:,:,filter_num], cmap='Spectral_r', vmax=5, vmin=0)
plt.colorbar()


activation[:,:,filter].mean()
activation[:,:,filter].std()
activation[:,:,filter].max()
plt.colorbar()


names = ["gm_159", 
         "gm_12",
         "gm_148",
         "gbm_159", 
         "gbm_12",
         "gbm_148",
         "met_159", 
         "met_12",
         "met_148"]



max_filter = 159
test = generate_pattern("activation_159", max_filter)
plt.imshow(test)


        
        
        
        
        
        
        

