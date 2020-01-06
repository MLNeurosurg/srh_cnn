#!/usr/bin/env python3

"""
Script to generate individual directories for each SRS mosaic's raw strips and move them to the newly made individual directories
root = root that contains the Study NIO directories
dest = directory to save all the moved raw files
"""
import os
import shutil
import numpy as np

def make_strip_directories(src, dest):
    
    # loop over all the directories in the root dir
    for dirs in os.listdir(src):

        # list of files in each subdirectory
        files = os.listdir(os.path.join(src, dirs))
        specimens = sorted(list(set([file.split("_")[1] for file in files])))

        nio = dirs.split("_")[-1]
        print(nio)
        for specimen in specimens:
            nio_dir = nio + "_" + specimen
            print(os.path.join(dest, nio_dir))
            try:
                os.makedirs(os.path.join(dest, nio_dir))
            except:
                pass

            for file in files:
                # check file sizes to ensure that are actually raw strips
                if np.round(os.stat(os.path.join(src, dirs, file)).st_size, decimals=-5) == np.round(12001080, decimals=-5) or np.round(os.stat(os.path.join(src, dirs, file)).st_size, decimals=-5) == np.round(8001080, decimals=-5) or np.round(os.stat(os.path.join(src, dirs, file)).st_size, decimals=-5) == np.round(20001080, decimals=-5):
                    if file.split("_")[1] == specimen:
                        try:
                            shutil.copy(src = os.path.join(src, dirs, file), dst = os.path.join(dest, nio_dir))
                        except:
                            continue

    # remove empty directories
    for dirs in os.listdir(dest):
        if len(os.listdir(os.path.join(dest, dirs))) == 0:
            os.rmdir(os.path.join(dest, dirs))
        else: 
            continue
                    

if __name__ == "__main__":

    root = ""
    dest = ""

    make_strip_directories(root, dest)


