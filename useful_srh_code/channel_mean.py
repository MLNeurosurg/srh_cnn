
import os
import random
from skimage import io
import numpy as np

from imageio import imread


img_rows = 300
img_cols = 300
img_channels = 3

def directory_importer(path, num_images, image = None, start_image = False):
    
    filelist = sorted(os.listdir(path))
    if not start_image:
        image = np.zeros((img_rows, img_cols, img_channels), dtype=float)

    counter = 0
    while counter < num_images:
        index = np.random.randint(len(filelist))
        random_image = imread(os.path.join(path, filelist[index])).astype(float)
        image += random_image
        counter += 1

    return image

if __name__ == "__main__":

    num_image = 10000
    total_dirs = 2
    accum_image = directory_importer("/home/todd/Desktop/SRH_genetics/srh_patches/patches/training_patches/training/IDHwt_gbm", num_images=num_image)
    image = directory_importer("/home/todd/Desktop/SRH_genetics/srh_patches/patches/training_patches/training/IDHmut_1p19q", num_images=num_image, image = accum_image, start_image = True)
    
    print(np.mean(image, axis=(0,1))/(num_image * total_dirs))
    print(np.std(image, axis = (0,1))/(num_image * total_dirs))



