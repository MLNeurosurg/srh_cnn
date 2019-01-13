

import os
import numpy as np
import shutil
import sys

# NIO069-4928_35_tile_016_20.tif
# Validation set
NIO_numbers_val = {
"ependymoma": ["NIO087-2310_4", "NIO087-2310_1"],
"glioblastoma": ["NIO016", "NIO082", "NIO108"],
"greymatter": ["NIO101"],
"lowgradeglioma": ["NIO030", "NIO063"],
"lymphoma": ["NIO033-7953_18", "NIO033-7953_15"],
"medulloblastoma": ["340-1564"],
"meningioma": ["NIO110", "NIO114", "NIO122"], 
"metastasis": ["NIO084", "NIO105", "NIO107"],
"nondiagnostic": [],
"pilocyticastrocytoma": ["NIO047"],
"pituitaryadenoma": ["NIO098", "NIO097"],
"pseudoprogression": ["NIO058-8765_1", "NIO058-8765_2"],
"whitematter": ["NIO105-1058_6", "NIO105-1058_10"]
}

# Testing set
# NIO_numbers_test = {
# "ependymoma": [],
# "glioblastoma": [],
# "greymatter": [],
# "lowgradeglioma": [],
# "lymphoma": [],
# "medulloblastoma": [],
# "meningioma": [],
# "metastasis": [],
# "nondiagnostic": [],
# "pilocyticastrocytoma": [],
# "pituitaryadenoma": [],
# "pseudoprogression": [],
# "whitematter": []
# }


def validation_set(validation_dict, validation_dir):
    class_list = list(validation_dict.keys())

    file_dict = {} # Initialize dicitionary with tumor class as key and each file (absolute path) is an element of a list
    for tumorclass in class_list:
        for root, dirs, files in os.walk("/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training"):
            for file in files:
                if (tumorclass in root) and ("tif" in file):
                    if tumorclass not in file_dict:
                        file_dict[tumorclass] = []
                    
                    file_dict[tumorclass].append(root + "/" + file)

    # Only search thru the tumor specific list
    for key_class, val_files in validation_dict.items():
        for val_file in val_files:

            for file in file_dict[key_class]: # index only into the tumor class
                if val_file in file:
                    shutil.move(src = file, dst = validation_dir + "/" + key_class)

    # Search the nondiagnostic class
    for _, val_files in validation_dict.items():
        for val_file in val_files:
            
            for file in file_dict["nondiagnostic"]: # index only into the tumor class
                if val_file in file:
                    shutil.move(src = file, dst = validation_dir + "/" + "nondiagnostic")

# def testing_set(testing_dict, testing_dir):
#     class_list = list(testing_dict.keys())

#     file_dict = {} # Initialize dicitionary with tumor class as key and each file (absolute path) is an element of a list
#     for tumorclass in class_list:
#         for root, dirs, files in os.walk("/home/orringer-lab/Desktop/Training_Images/cnn_tiles_training"):
#             for file in files:
#                 if (tumorclass in root) and ("tif" in file):
#                     file_dict[tumorclass] = root + "/" + file

#     # Only search thru the tumor specific list
#     for key_class, val_files in testing_dict.items():
#         for file in file_dict[key_class]: # index only into the tumor class
#             if (val_files in file):
#                 shutil.move(src = val, dst = testing_dir + "/" + key_class)
#     # Search the nondiagnostic class
#     for _ , val_files in testing_dict.items():
#         for file in file_dict["nondiagnostic"]: # must include nondiagnostic class
#             if (val_files in file):
#                 shutil.move(src = val, dst = testing_dir + "/" + "nondiagnostic")


if __name__ == '__main__':

    validation_set(NIO_numbers_val, "/home/orringer-lab/Desktop/Training_Images/cnn_tiles_validation")
    # testing_set(NIO_numbers_test, DIRHERE)
