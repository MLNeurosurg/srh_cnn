# TODO
# close the dicom files?

import os
import sys
import pydicom as dicom
import re
import numpy as np
from skimage.io import imread, imsave

folder_dir = ""
image_dict = {}

def sort_all_scans():
    for file in file_list:
        # scan_number = int(file[8:12])
        scan_number = int(file.split("_")[1])
        if scan_number not in image_dict:
            image_dict[scan_number] = []

        image_dict[scan_number].append(file)

def array_is_descending_gradient(pixel_array, y_size):
    for y in range(int(y_size / 25) - 2):
        for x in range(38):
            if (np.average(pixel_array[(y*25):(y+1)*25,(x*25):(x+1)*25]) < np.average(pixel_array[(y+1)*25:(y+2)*25,(x+1)*25:(x+2)*25])):
                return False
    return True

def array_is_ascending_gradient(pixel_array, y_size):
    for y in range(int(y_size / 25) - 2):
        for x in range(38):
            if (np.average(pixel_array[(y*25):(y+1)*25,(x*25):(x+1)*25]) > np.average(pixel_array[(y+1)*25:(y+2)*25,(x+1)*25:(x+2)*25])):
                return False
    return True

def array_is_noise(pixel_array):
    if (pixel_array.size - np.count_nonzero(pixel_array)) > ((pixel_array.shape[0] * pixel_array.shape[1]) * 0.20):
        return True

    histogram = np.histogram(pixel_array)[0]
    if (histogram[0] != 0) and (histogram[1] == 0) and (histogram[8] == 0) and (histogram[9] != 0):
        return True
    
    return False

def sort_SRH_scans(scan_number, SRH_rows, SRH_columns):
    max_image_number = max([int(f.split("_")[2].split(".")[0]) for f in image_dict[scan_number]])
    # max_image_number = max([int(f[13:17]) for f in image_dict[scan_number]])
    unique_strip_count = SRH_columns / 1000
    num_channels_per_strip = 2
    strip_dict = {} 

    # Assertion that the SRH image is square
    assert((SRH_rows / 1000) == unique_strip_count)
    
    # Get number of discovered strips
    strip_count = 0
    for i in range(1, max_image_number + 1):
        # im = dicom.read_file(os.path.join(folder_dir, "ScanImg_" + str(scan_number).zfill(4) + "_" + str(i).zfill(4) + ".dcm"))
        
        im = dicom.read_file(os.path.join(folder_dir, "img_" + str(scan_number) + "_" + str(i) + ".dcm"))

        if (im.Columns == 1000) and (im.Rows == SRH_rows) and (im.PhotometricInterpretation == "MONOCHROME2"):
            strip_count += 1
    
    if ((strip_count / 4) == unique_strip_count) and ((strip_count % 4) == 0):
        num_channels_per_strip = 4
    elif ((strip_count / 2) == unique_strip_count) and ((strip_count % 2) == 0):
        num_channels_per_strip = 2
    else:
        raise Exception("Found " + str(strip_count) + " strips. Expected: " + str(unique_strip_count))

    if num_channels_per_strip == 4:
        raise Exception("Found 4 channels of info for an SRH file")

    current_strip = 0
    for i in range(1, max_image_number + 1):
        # dcm_filename = "ScanImg_" + str(scan_number).zfill(4) + "_" + str(i).zfill(4) + ".dcm"
        
        dcm_filename = "img_" + str(scan_number) + "_" + str(i) + ".dcm"

        im = dicom.read_file(os.path.join(folder_dir, dcm_filename))
        if (im.Columns == 1000) and (im.Rows == SRH_rows) and (im.PhotometricInterpretation == "MONOCHROME2"):
            current_unique_strip = int(current_strip / num_channels_per_strip)
            if current_unique_strip not in strip_dict:
                strip_dict[current_unique_strip] = {'CH2':'', 'CH3':'', 'DNA':'', 'Water':''}
            
            CH2_found = (strip_dict[current_unique_strip]['CH2'] != '')
            CH3_found = (strip_dict[current_unique_strip]['CH3'] != '')

            if (CH2_found and not CH3_found):
                strip_dict[current_unique_strip]['CH3'] = dcm_filename
            if (not CH2_found):
                strip_dict[current_unique_strip]['CH2'] = dcm_filename

            current_strip += 1
    

    return strip_dict


def join_strips(root_dicom, strip_dict):
    mosaic_image = np.empty((root_dicom.Rows, root_dicom.Columns, 3), dtype=np.uint16)
    
    for strip in strip_dict:
        im_ch2 = dicom.read_file(os.path.join(folder_dir, strip_dict[strip]['CH2']))
        im_ch3 = dicom.read_file(os.path.join(folder_dir, strip_dict[strip]['CH3']))
        
        # ch2_float_max = float(im_ch2.pixel_array.max())
        # ch3_float_max = float(im_ch3.pixel_array.max())

        # ch2_array = ((im_ch2.pixel_array.astype('float') / ch2_float_max) * 255.0) * 1.5
        # ch3_array = ((im_ch3.pixel_array.astype('float') / ch3_float_max) * 255.0) * 0.75
        # subtracted_array = np.subtract(ch3_array, ch2_array)
        
        # # # I added these lines
        # subtracted_array_max = float(subtracted_array.max())  
        # subtracted_array = ((subtracted_array.astype('float') / subtracted_array_max) * 255.0)
        
        ### Changed these lines
        ch2_array = im_ch2.pixel_array.astype('int16')  
        ch3_array = im_ch3.pixel_array.astype('int16')
        subtracted_array = np.subtract(ch3_array, ch2_array)
        subtracted_array[subtracted_array < 0] = 0.0

        strip_size = im_ch2.Columns
        mosaic_image[:, (strip * strip_size):(strip+1) * strip_size, 0] = subtracted_array.astype('int16')
        mosaic_image[:, (strip * strip_size):(strip+1) * strip_size, 1] = ch2_array.astype('int16')
        mosaic_image[:, (strip * strip_size):(strip+1) * strip_size, 2] = ch3_array.astype('int16')
        ####

    return mosaic_image


def sort_4ch_scans(scan_number, root_rows, root_columns):
    # max_image_number = max([int(f[13:17]) for f in image_dict[scan_number]])
    max_image_number = max([int(f.split("_")[2].split(".")[0]) for f in image_dict[scan_number]])
    strip_dict = {} 
    files = []

    for dicom_file in image_dict[scan_number]:
        im = dicom.read_file(os.path.join(folder_dir, dicom_file))
        if (im.BitsStored == 16) and (im.Rows == root_rows) and (im.Columns == root_columns):
            if array_is_descending_gradient(im.pixel_array, im.Rows):
                continue

            if array_is_ascending_gradient(im.pixel_array, im.Rows):
                continue

            if array_is_noise(im.pixel_array):
                continue

            files.append(dicom_file)

    # ch2_file = "ScanImg_" + str(scan_number).zfill(4) + "_" + str(1).zfill(4) + ".dcm"
    # ch3_file = "ScanImg_" + str(scan_number).zfill(4) + "_" + str(7).zfill(4) + ".dcm"
    # dna_file = "ScanImg_" + str(scan_number).zfill(4) + "_" + str(13).zfill(4) + ".dcm"
    # water_file = "ScanImg_" + str(scan_number).zfill(4) + "_" + str(19).zfill(4) + ".dcm"
    
    ch2_file = "img_" + str(scan_number) + "_" + str(1) + ".dcm"
    ch3_file = "img_" + str(scan_number) + "_" + str(7) + ".dcm"
    dna_file = "img_" + str(scan_number) + "_" + str(13) + ".dcm"
    water_file = "img_" + str(scan_number) + "_" + str(19) + ".dcm"

    if (ch2_file not in files) or (ch3_file not in files) or (dna_file not in files) or (water_file not in files):
        raise Exception('All 4CH files not found in scan')

    strip_dict[0] = {'CH2':'', 'CH3':'', 'DNA':'', 'Water':''}

    strip_dict[0]['CH2'] = ch2_file
    strip_dict[0]['CH3'] = ch3_file
    strip_dict[0]['DNA'] = dna_file
    strip_dict[0]['Water'] = water_file

    return strip_dict

def determine_imaging_modality(scan_number):
    for dicom_file in image_dict[scan_number]:
        im = dicom.read_file(os.path.join(folder_dir, dicom_file))
        if (im.PhotometricInterpretation == "RGB") and (im.BitsStored == 8):
            return ('SRH', dicom_file)

    root_4ch_file = None
    for dicom_file in image_dict[scan_number]:
        im = dicom.read_file(os.path.join(folder_dir, dicom_file))
        if (im.PhotometricInterpretation == "RGB") and (im.BitsStored == 16):
            root_4ch_file = dicom_file
    
    if (root_4ch_file != None):
        channel_count = 0
        root_dicom = dicom.read_file(os.path.join(folder_dir, root_4ch_file))

        for dicom_file in image_dict[scan_number]:
            im = dicom.read_file(os.path.join(folder_dir, dicom_file))
            if (im.PhotometricInterpretation == "MONOCHROME2") and (im.BitsStored == 16) and (im.Rows == root_dicom.Rows) and (im.Columns == root_dicom.Columns):
                if array_is_descending_gradient(im.pixel_array, im.Rows):
                    continue

                if array_is_ascending_gradient(im.pixel_array, im.Rows):
                    continue

                if array_is_noise(im.pixel_array):
                    continue

                channel_count += 1

        if (channel_count == 7):
            # 4 channel images and 3 transmission(?) images
            return ('4CH', root_4ch_file)
        else:
            return ('Non SRH/4CH', channel_count)

    else:
        return (None, None)

def save_tiled_mosaic(mosaic_image, prefix):
    y_size = mosaic_image.shape[0]
    x_size = mosaic_image.shape[1]

    y_count = int(y_size / 1000)
    x_count = int(x_size / 1000)


    for y in range(y_count):
        for x in range(x_count):
            imsave(prefix + "_tile_" + str((y * y_count) + (x) + 1).zfill(3) + ".tif", mosaic_image[(y*1000):(y+1)*1000, (x*1000):(x+1)*1000])
    

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please provide a folder location")
        print("Example: python DCM_to_PNG.py /Users/Balaji/NIO_004")
        sys.exit(1)

    folder_dir = sys.argv[1]

    file_list = os.listdir(folder_dir)
    file_list = [f for f in file_list if (".dcm" in f)]

    sort_all_scans()
    
    for scan_number in image_dict:
        print("scan number: " + str(scan_number)),

        #Determine if SRH or 4-channel GBM
        imaging_modality = determine_imaging_modality(scan_number)

        mosaic_image = None
        if (imaging_modality[0] == 'SRH'):
            print(" SRH")
            SRH_dicom = dicom.read_file(os.path.join(folder_dir, imaging_modality[1]))
            strip_dict = sort_SRH_scans(scan_number, SRH_dicom.Rows, SRH_dicom.Columns)
            mosaic_image = join_strips(SRH_dicom, strip_dict)
            save_tiled_mosaic(mosaic_image, "NIO" + re.sub("\D", "", SRH_dicom.AccessionNumber) + "-" + str(SRH_dicom.PatientID[-4:]) + "_" + str(scan_number))
        elif (imaging_modality[0] == '4CH'):
            print(" 4CH")
            root_dicom = dicom.read_file(os.path.join(folder_dir, imaging_modality[1]))
            strip_dict = sort_4ch_scans(scan_number, root_dicom.Rows, root_dicom.Columns)
            mosaic_image = join_strips(root_dicom, strip_dict)
            save_tiled_mosaic(mosaic_image, "NIO" + re.sub("\D", "", root_dicom.AccessionNumber) + "-" + str(root_dicom.PatientID[-4:]) + "_" + str(scan_number))
        else:
            print(imaging_modality)




