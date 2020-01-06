#!/usr/bin/env python3

# standard python
import os
import pydicom as dicom
import numpy as np

from preprocessing.preprocess import *
from preprocessing.registration import *


def import_srh_dicom(image_dir, flatten = False, filter_type = "gaussian"):
    """
    image_dir must contain only the image strips for a single mosaic
    First file = CH2
    Second file = CH3
    """
    files = [file for file in os.listdir(image_dir) if ("dcm" in file)]
    image_num = []
    for file in files:
        _ ,_ , num_str = file.split(sep=("_"))
        image_num.append(int(num_str.split(sep=".")[0]))
    
    # sort the strips based on the image image number, NOTE: Assumes CH2 is the first channel captured
    sorted_files = [name for name, _ in sorted(zip(files, image_num), key=lambda pair: pair[1])]

    # read in every other file
    CH2_files = sorted_files[::2]
    CH3_files = sorted_files[1::2]
    
    # import the first image to get dicom specs
    CH3 = dicom.read_file(os.path.join(image_dir, CH3_files[0])).pixel_array.astype(float)
    CH2 = dicom.read_file(os.path.join(image_dir, CH2_files[0])).pixel_array.astype(float)

    def import_array(filelist, first_strip):
        """
        Iteratively concatenate each strip columnwise
        """
        first_strip_copy = np.copy(first_strip)
        for file in filelist[1:]: 
            try:
                strip = dicom.read_file(os.path.join(image_dir, file)).pixel_array.astype(float)
                first_strip = np.concatenate((first_strip, strip), axis = 1)
            # field flat here if needed
            except: # exception to handle error in importing some bad dicom files
                first_strip = np.concatenate((first_strip, first_strip_copy), axis = 1)
                print("Stip needed to be copied due to BAD dicom file.")
        return first_strip

    # import each channel
    CH2 = import_array(CH2_files, CH2)
    CH3 = import_array(CH3_files, CH3)
    
    # flatten image
    if flatten:
        print("FLATTENING")
        CH2 = field_flattening(CH2, filter_type = filter_type)
        CH3 = field_flattening(CH3, filter_type = filter_type)

    # register the images with respect to each strip
    image_reg = register_mosaic(CH2, CH3)

    # channel subtraction
    subtracted_array = np.subtract(image_reg[:,:,1], image_reg[:,:,0]) # CH3 minus CH2
    subtracted_array[subtracted_array < 0] = 0.0 # negative values set to zero

    # concatentate the postprocessed images
    dcm_stack = np.zeros((CH2.shape[0], CH2.shape[1], 3), dtype=np.float)
    dcm_stack[:, :, 0] = subtracted_array
    dcm_stack[:, :, 1] = image_reg[:,:,0]
    dcm_stack[:, :, 2] = image_reg[:,:,1]

    return dcm_stack.astype(np.float)

def import_renorm_dicom(image_dir):
    """
    image_dir must contain only the image strips for a single mosaic
    First file = CH2
    Second file = CH3
    """
    files = [file for file in os.listdir(image_dir) if ("dcm" in file)]
    image_num = []
    for file in files:
        _ ,_ , num_str = file.split(sep=("_"))
        image_num.append(int(num_str.split(sep=".")[0]))
    
    # sort the strips based on the image image number, NOTE: Assumes CH2 is the first channel captured
    sorted_files = [name for name, _ in sorted(zip(files, image_num), key=lambda pair: pair[1])]

    # read in every other file
    CH2_files = sorted_files[::2]
    CH3_files = sorted_files[1::2]
    
    # import the first image to get dicom specs
    CH3 = dicom.read_file(os.path.join(image_dir, CH3_files[0])).pixel_array.astype(float)
    CH2 = dicom.read_file(os.path.join(image_dir, CH2_files[0])).pixel_array.astype(float)
    
    def import_array(filelist, first_strip):
        """
        Iteratively concatenate each strip columnwise
        """
        for file in filelist[1:]: 
            strip = dicom.read_file(os.path.join(image_dir, file)).pixel_array.astype(float)
            first_strip = np.concatenate((first_strip, strip), axis = 1)
        return first_strip

    # import each channel
    CH2 = import_array(CH2_files, CH2)
    CH3 = import_array(CH3_files, CH3)
    
    # rescale the channels
    CH2 = min_max_rescaling(CH2) * 255
    CH3 = min_max_rescaling(CH3) * 255

    image_reg = register_mosaic(CH2, CH3)

    subtracted_array = np.subtract(CH3, CH2)
    subtracted_array[subtracted_array < 0] = 0.0 # negative values set to zero

    dcm_stack = np.zeros((CH2.shape[0], CH2.shape[1], 3), dtype=float)
    dcm_stack[:, :, 0] = subtracted_array
    dcm_stack[:, :, 1] = CH2
    dcm_stack[:, :, 2] = CH3

    return dcm_stack.astype(np.uint8)


if __name__ == '__main__':
	pass