'''
Splitting white and grey matter

strategy, split INV and NIO images

'''

from skimage.io import imread
import os
import numpy as np
from scipy.stats import kurtosis, skew
import shutil

os.chdir("/home/orringer-lab/Desktop/TIF_NIO_testing/normal")
file_dict = {}
filelist = os.listdir()

for file in filelist:
    file_dict[file] = imread(file)

def return_channels(img):
    red = img[:,:,0]
    CH2 = img[:,:,1]
    CH3 = img[:,:,2]
    return red, CH2, CH3

for file in filelist:
    print(file)
    img = imread(file)
    _, CH2, CH3 = return_channels(img)
    if skew(CH2.flatten()) < 1.5:
        shutil.move(file, dst="/home/orringer-lab/Desktop/TIF_NIO_testing/whitematter")
    else:
        shutil.move(file, dst="/home/orringer-lab/Desktop/TIF_NIO_testing/greymatter")


