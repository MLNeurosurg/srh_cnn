#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:11:40 2017

@author: orringer-lab
"""

import os
import random
from scipy.misc import imresize
from skimage import io
import numpy as np

#class_names = [
# 'glioblastoma',
# 'lowgradeglioma',
# 'meningioma',
# 'metastasis',
# 'nondiagnostic',
# 'normal', 
# 'pituitaryadenoma']

img_rows = 300
img_cols = 300
img_channels = 3

def channel_mean(images):
    num_images = len(images)
    CH3minusCH2 = 0
    CH2 = 0
    CH3 = 0
    for i in images:
        CH3minusCH2 += i[:,:,0].mean()
        CH2 += i[:,:,1].mean()
        CH3 += i[:,:,2].mean()

    return(CH3minusCH2/num_images, CH2/num_images, CH3/num_images)


def getRandomFile(path):
  """
  Returns a random filename, chosen among the files of the given path.
  """
  files = os.listdir(path)
  index = random.randrange(0, len(files))
  return(files[index])

def SRH_import():
    # Ependymoma
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/ependymoma')
    random_ependy_files = []
    ependymoma_rand_list = []
    while len(random_ependy_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/ependymoma')
        random_ependy_files.append(new_index)
    for i in random_ependy_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        ependymoma_rand_list.append(im)
    ependymoma_rand_list = np.asarray(ependymoma_rand_list)

    # GBM
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/glioblastoma')
    random_gbm_files = []
    gbm_rand_list = []
    while len(random_gbm_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/glioblastoma')
        random_gbm_files.append(new_index)
    for i in random_gbm_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        gbm_rand_list.append(im)
    gbm_rand_list = np.asarray(gbm_rand_list)
    
    # LGG
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/lowgradeglioma')
    random_lgg_files = []
    lgg_rand_list = []
    while len(random_lgg_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/lowgradeglioma')
        random_lgg_files.append(new_index)
    for i in random_lgg_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        lgg_rand_list.append(im)
    lgg_rand_list = np.asarray(lgg_rand_list)

    # lymphoma
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/lymphoma')
    random_lymphoma_files = []
    lymphoma_rand_list = []
    while len(random_lymphoma_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/lymphoma')
        random_lymphoma_files.append(new_index)
    for i in random_lymphoma_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        lymphoma_rand_list.append(im)
    lymphoma_rand_list = np.asarray(lymphoma_rand_list)

    # meningioma
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/meningioma')
    random_medullo_files = []
    medullo_rand_list = []
    while len(random_medullo_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/meningioma')
        random_medullo_files.append(new_index)
    for i in random_medullo_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        medullo_rand_list.append(im)
    medullo_rand_list = np.asarray(medullo_rand_list)
    
    # meningioma
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/meningioma')
    random_mening_files = []
    mening_rand_list = []
    while len(random_mening_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/meningioma')
        random_mening_files.append(new_index)
    for i in random_mening_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        mening_rand_list.append(im)
    mening_rand_list = np.asarray(mening_rand_list) 
        
    # metastasis
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/metastasis')
    random_met_files = []
    met_rand_list = []
    while len(random_met_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/metastasis')
        random_met_files.append(new_index)
    for i in random_met_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        met_rand_list.append(im)
    met_rand_list = np.asarray(met_rand_list)   
    
    # nondiagnostic
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/nondiagnostic')
    random_nondiag_files = []
    nondig_rand_list = []
    while len(random_nondiag_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/nondiagnostic')
        random_nondiag_files.append(new_index)
    for i in random_nondiag_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        nondig_rand_list.append(im)
    nondig_rand_list = np.asarray(nondig_rand_list)   
        
    # greymatter
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/greymatter')
    random_normal_files = []
    normal_rand_list = []
    while len(random_normal_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/greymatter')
        random_normal_files.append(new_index)
    for i in random_normal_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        normal_rand_list.append(im)
    normal_rand_list = np.asarray(normal_rand_list) 

    # whitematter
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/whitematter')
    random_whitematter_files = []
    whitematter_rand_list = []
    while len(random_whitematter_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/whitematter')
        random_whitematter_files.append(new_index)
    for i in random_whitematter_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        whitematter_rand_list.append(im)
    whitematter_rand_list = np.asarray(whitematter_rand_list) 

    # Pilocytic astrocytoma
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/pilocyticastrocytoma')
    random_pilo_files = []
    pilo_rand_list = []
    while len(random_pilo_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/pilocyticastrocytoma')
        random_pilo_files.append(new_index)
    for i in random_pilo_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        pilo_rand_list.append(im)
    pilo_rand_list = np.asarray(pilo_rand_list)

    # pituitary adenoma
    os.chdir('/home/todd/Desktop/CNN_Images/inv_training_tiles/pituitaryadenoma')
    random_pit_files = []
    pit_rand_list = []
    while len(random_pit_files) < 3000:
        new_index = getRandomFile('/home/todd/Desktop/CNN_Images/inv_training_tiles/pituitaryadenoma')
        random_pit_files.append(new_index)
    for i in random_pit_files:
        im = imresize(io.imread(i),(img_rows,img_cols))
        pit_rand_list.append(im)
    pit_rand_list = np.asarray(pit_rand_list)  
    
    
    # Concatenate images for single dataset for centering and normalization 
    centering_image_data = np.concatenate((ependymoma_rand_list, gbm_rand_list, lgg_rand_list, whitematter_rand_list, lymphoma_rand_list, medullo_rand_list, mening_rand_list, met_rand_list, nondig_rand_list, normal_rand_list, pilo_rand_list, pit_rand_list))
    return(centering_image_data)

if __name__ == "__main__":
    centering_image_data = SRH_import()
    foo = channel_mean(centering_image_data)
    print(foo)

