
# standard python
import os
import numpy as np

# plotting
import matplotlib.pyplot as plt
from cv2 import GaussianBlur, medianBlur

IMAGE_SIZE = 300

def return_channels(array):
    """
    Helper function to return channels
    """
    return array[:,:,0], array[:,:,1], array[:,:,2]

def min_max_rescaling(array, percentile_clip = 3):
    p_low, p_high = np.percentile(array, (percentile_clip, 100 - percentile_clip))
    array = array.clip(min = p_low, max = p_high)
    img = (array - p_low)/(p_high - p_low)
    return img

def channel_rescaling(array):
    DNA, CH2, CH3 = return_channels(array.astype('float'))
    img = np.empty((array.shape[0], array.shape[1], 3), dtype='float')

    img[:,:,0] = min_max_rescaling(DNA)
    img[:,:,1] = min_max_rescaling(CH2)
    img[:,:,2] = min_max_rescaling(CH3)

    img *= 255
    
    return img

def field_flattening(image, filter_type = "gaussian", ksize = 31, sigma = 101):
    """
    Function that will field flatten a single channel image
    """
    if filter_type == "gaussian":
        blurred_image = GaussianBlur(image.astype(np.uint8), ksize = (ksize, ksize), sigmaX = sigma)
    if filter_type == "median":
        blurred_image = medianBlur(image.astype(np.uint8), ksize = ksize)

    flat_image = np.divide(image.astype(float), blurred_image.astype(float))

    flat_image[np.isnan(flat_image)] = 0
    flat_image = (min_max_rescaling(flat_image)) * 255

    return flat_image

def field_flattening_CH2(image, filter_type = "gaussian", ksize = 51, sigma = 101):
    """
    Function that will field flatten a single CH2 channel image
    """
    if filter_type == "gaussian":
        blurred_image = GaussianBlur(image.astype(np.uint8), ksize = (ksize, ksize), sigmaX = sigma)
    if filter_type == "median":
        blurred_image = medianBlur(image.astype(np.uint8), ksize = ksize)

    flat_image = np.divide(image.astype(float), blurred_image.astype(float) * 1.25)

    flat_image[np.isnan(flat_image)] = 0
    flat_image = (min_max_rescaling(flat_image)) * 255

    return flat_image

def cnn_preprocessing(image):
    """
    Channel-wise means calculated over NIO dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

def random_crop(image, crop_size = IMAGE_SIZE):
    '''
    Generate a random crop from a SRH mosaic
    '''
    edge = int(image.shape[0])
    random_x = np.random.randint(0, int(edge - crop_size))
    random_y = np.random.randint(0, int(edge - crop_size))
    
    crop = image[random_x:random_x + crop_size, random_y:random_y + crop_size, :]
    
    return crop


if __name__ == '__main__':
	pass