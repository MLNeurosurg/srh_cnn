

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:42:45 2017

@author: orringer-lab
"""

'''
Keras-vis ConvNet filter visualization

Most of the methods for this library is from 
For a full description of saliency, see the paper:
[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps] (https://arxiv.org/pdf/1312.6034v2.pdf)

'''
#################################################The code that actually works


# Activation Maximization using example script
# SALIENCY MAP for SRH CNN
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from keras.models import load_model
from vis.visualization.activation_maximization import visualize_activation
from vis.visualization.saliency import visualize_saliency
from vis.utils import utils
from keras import activations

'''
visualize_activation(model, layer_idx, filter_indices=None, seed_input=None, input_range=(0, 255), \
    backprop_modifier=None, grad_modifier=None, act_max_weight=1, lp_norm_weight=10, \
    tv_weight=10, **optimizer_params)
'''


# List layers in dictionary
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'activation_4')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

#############
from vis.utils import utils
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (18, 6)

class_indices = {'ependymoma': 0,
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
 'whitematter': 12}

img_cols, img_rows = 300, 300

img_mening = utils.load_img('/home/todd/Desktop/CNN_Images/nio_validation_tiles/metastasis/NIO084-9521_1_tile_001_0.tif', target_size=(300, 300))
img_met = utils.load_img('/home/todd/Desktop/CNN_Images/nio_validation_tiles/metastasis/NIO084-9521_1_tile_001_4.tif', target_size=(300, 300)) 

f, ax = plt.subplots(1, 2)
ax[0].imshow(img_met)
ax[1].imshow(img_mening)
plt.show()

###############
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

heatmap = visualize_saliency(model, layer_idx, filter_indices=1, seed_input=img_mening)
plt.imshow(heatmap)
plt.imshow(overlay(img_mening, heatmap))


#f, ax = plt.subplots(1, 2)
#for i, img in enumerate([img_gbm, img_mening]):    
#    # 20 is the imagenet index corresponding to `ouzel`
#    heatmap = visualize_saliency(model, layer_idx, filter_indices=2, seed_input=img)
#    # Lets overlay the heatmap onto original image.    
#    ax[i].imshow(overlay(img, heatmap))
    
###############
'''
To use guided saliency, we need to set backprop_modifier='guided'. For rectified saliency or deconv saliency, use backprop_modifier='relu'.
'''
for modifier in ['guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle(modifier)
    for i, img in enumerate([heatmap_gbm, img_pit]):    
        # 20 is the imagenet index corresponding to `ouzel`
        heatmap = visualize_saliency(model, layer_idx, filter_indices=1, 
                                     seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.    
        ax[i].imshow(overlay(img, heatmap))


heatmap_pit = visualize_saliency(model, layer_idx, filter_indices=6, seed_input=img_pit, backprop_modifier='guided')
imshow(overlay(img_pit, heatmap_pit))


##############
'''
guided grad-CAM wins again. It far less noisy than other options.
'''
from vis.visualization import visualize_cam

for modifier in [None, 'guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img3]):    
        # 20 is the imagenet index corresponding to `ouzel`
        heatmap = visualize_cam(model, layer_idx, filter_indices=3, 
                                seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.    
        ax[i].imshow(overlay(img, heatmap))


{'glioblastoma': 0,
 'lowgradeglioma': 1,
 'meningioma': 2,
 'metastasis': 3,
 'nondiagnostic': 4,
 'normal': 5,
 'pituitaryadenoma': 6}
