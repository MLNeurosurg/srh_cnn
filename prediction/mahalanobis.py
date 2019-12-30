

import os
import sys
import time
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from keras.models import load_model
from pandas import DataFrame
from collections import defaultdict

from keras.utils import multi_gpu_model
from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D 
# Open-source models
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from scipy.spatial.distance import mahalanobis
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame

from preprocessing.preprocess import cnn_preprocessing
from preprocessing.patch_generator import patch_generator

img_rows = 300
img_cols = 300
img_channels = 3
total_classes = 14
class_names = ['ependymoma',
 'greymatter',
 'glioblastoma',
 'lowgradeglioma',
 'lymphoma',
 'medulloblastoma',
 'meningioma',
 'metastasis',
 'nondiagnostic',
 'normal',
 'pilocyticastrocytoma',
 'pituitaryadenoma',
 'pseudoprogression',
 'schwannoma',
 'whitematter']

layer_outputs = [
"conv2d_159",
"conv2d_195",
"conv2d_199",
"conv2d_203",
"average_pool",
"dense",
"softmax"]

def cnn_preprocessing(image):
    """
    Channel-wise means calculated over NIO dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

def find_pair_factors_for_CNN(x):
    """
    Function to match batch size and iterations for the validation generator
    """
    pairs = []
    for i in range(2, 150):
        test = x/i
        if i * int(test) == x:
            pairs.append((i, int(test)))
    best_pair = pairs[-1]
    assert len(pairs) >= 1, "No pairs found"
    print(best_pair)
    return best_pair

def validation_batch_steps(directory):
    counter = 0
    for roots, dirs, files in os.walk(directory):
        for file in files:
            counter += 1
    return find_pair_factors_for_CNN(counter)

def mahalanobis_model(parent_model_path):
    """
    Output layer index is from the TOP LAYER BACK
    """
    model = load_model(parent_model_path)
    mahalanobis_model = Model(input=model.inputs, outputs=[
    model.get_layer("conv2d_159").output,
    model.get_layer("conv2d_195").output,
    model.get_layer("conv2d_199").output,
    model.get_layer("conv2d_203").output,
    model.layers[-7].output,
    model.layers[-4].output,
    model.output])

    return mahalanobis_model

def average_pool_batch(model_output):
    
    # check if just a 2D array
    if len(model_output.shape) == 2: 
        return model_output
    
    else:
        num_patches = model_output.shape[0]
        ave_pool_vect = np.zeros((model_output.shape[0], model_output.shape[-1]))
        for i in range(num_patches):
            print(i)
            ave_pooled = model_output[i,:,:,:].mean(axis = (0,1))
            ave_pool_vect[i,:] = ave_pooled
        return ave_pool_vect

def average_pool_online(array):
    
    # check if just a  2D array
    if len(array.shape) == 2: 
        return array
    else:
        return array.mean(axis = (1,2))

def mu_sigma(model_output, train_generator):
    """
    Fuction takes a model output and train generator and return a tuple of 1) class specific dictionary of mean layer activation values and 2) covariance matrix
    """
    targets = train_generator.classes # use for indexing
    target_classes = list(train_generator.class_indices.values()) # list of classes
    target_mu_dict = {}
    for target_class in target_classes:
        target_mu_dict[target_class] = model_output[targets == target_class].mean(axis = 0)

    sigma = np.cov(model_output, rowvar=False)
    return target_mu_dict, sigma


class Class_conditional_dicts(object):
    def __init__(self, layer_list):
        
        self.layer_list = layer_list
        self.layer_dict = defaultdict(tuple)

    def populate_mu_sigmas(self, model_ouptut, train_generator):
        
        for i, layer in enumerate(self.layer_list):
            self.layer_dict[layer] = mu_sigma(average_pool_batch(model_ouptut[i]), train_generator)
            

def feedforward(patch_array, model):
    """
    Function to perform a forward pass, with preprocessing on all the patches generated above, outputs 
    """
    num_patches = patch_array.shape[0]

    conv2d_159 = np.zeros((1, 256), dtype=float)
    conv2d_195 = np.zeros((1, 256), dtype=float)
    conv2d_199 = np.zeros((1, 256), dtype=float)
    conv2d_203 = np.zeros((1, 256), dtype=float)
    average_pool = np.zeros((1, 1536), dtype=float)
    dense = np.zeros((1, 20), dtype=float)
    softmax = np.zeros((1, 14), dtype=float)

    nondiag_count = 0
    for i in range(num_patches):
        print(i)
        patch = cnn_preprocessing(patch_array[i,:,:,:])
        conv2d_159_iter, conv2d_195_iter, conv2d_199_iter, conv2d_203_iter, average_pool_iter, dense_iter, softmax_iter = model.predict(patch[None,:,:,:], batch_size = 1)

        if softmax_iter.argmax() == 8: # nondiagnostic class
            print("nondiagostic")
            nondiag_count += 1 
        else:
            conv2d_159 = np.concatenate((conv2d_159, average_pool_online(conv2d_159_iter)), axis = 0)
            conv2d_195 = np.concatenate((conv2d_195, average_pool_online(conv2d_195_iter)), axis = 0)
            conv2d_199 = np.concatenate((conv2d_199, average_pool_online(conv2d_199_iter)), axis = 0)
            conv2d_203 = np.concatenate((conv2d_203, average_pool_online(conv2d_203_iter)), axis = 0)
            average_pool = np.concatenate((average_pool, average_pool_iter), axis = 0)
            dense = np.concatenate((dense, dense_iter), axis = 0)
            softmax = np.concatenate((softmax, softmax_iter), axis = 0)

    print(nondiag_count)
    return conv2d_159[1:,:], conv2d_195[1:,:], conv2d_199[1:,:], conv2d_203[1:,:], average_pool[1:,:], dense[1:,:], softmax[1:, :]


def mahalanobis_score(test_ouput, mu_vector, sigma):

    assert test_ouput.shape[0] == sigma.shape[0]

    precision_matrix = np.linalg.inv(sigma)

    mah_score = 10E8
    target_mah = ""
    for target, mu_vector in mu_vector.items():
        
        score = mahalanobis(mu_vector, test_ouput, precision_matrix)
        
        if score < mah_score:
            mah_score = score
            target_mah = target
        else:
            continue
    return (mah_score, target_mah)

def mahalanobis_stats(representaiton_output, mu_vector_dict, sigma):
    scores = []
    for i in representaiton_output:
        score, post = mahalanobis_score(i, mu_vector_dict, sigma)
        scores.append(score)
        print(score)

    print("Mean score: " + str(np.mean(scores)))
    print("Std score: " + str(np.std(scores)))
    print("90th percentile score: " + str(np.percentile(scores, 90)))
    return (np.mean(scores), np.std(scores), np.percentile(scores, 90))

def mahalanobis_score_ensemble(class_cond_object, test_output):

    stats = {}
    score = 0
    mu_sigma_dicts = class_cond_object.layer_dict

    for i, layer in enumerate(class_cond_object.layer_list):
        mu, sigma = mu_sigma_dicts[layer][0], mu_sigma_dicts[layer][1]
        stats[layer] = mahalanobis_stats(test_output[i], mu, sigma)
    
    return stats

def directory_iterator(root):
    mosaic_dirs = os.listdir(root)
    mosaic_dirs = sorted(mosaic_dirs)

    print(mosaic_dirs)
    
    counter = 0
    mosaic_dict = {}
    for mosaic_dir in mosaic_dirs:
        counter += 1

        mosaic = import_raw_dicom(os.path.join(root, mosaic_dir))
        patches = patch_generator(mosaic)
        inference_mah = feedforward(patches, model=output_model)
        score_dict = mahalanobis_score_ensemble(maha_class, inference_mah)

        mosaic_dict[mosaic_dir] = score_dict
        print(counter)

    return mosaic_dict

def export_mahalanobis_scores(mosaic_dict, layer_outputs, metric = "mean"):

    df_dict = {}
    for layer in layer_outputs:
        df_dict[layer] = []
    df_dict["mosaics"] = []

    for mosaic, mahalanobis_score_dict in mosaic_dict.items():
        for layers, mahalanobis_scores in mahalanobis_score_dict.items():
            if metric == "mean":
                df_dict[layers].append(mahalanobis_score_dict[layers][0])

            if metric == "90th_percentile":
                df_dict[layers].append(mahalanobis_score_dict[layers][2])

        df_dict["mosaics"].append(mosaic)

    df = DataFrame(df_dict)
    df.to_excel("rare_cases_"+ str(metric) + "_mahalanobis.xlsx")


def check_means(mosaic_dict):
    for mosaic, scores in mosaic_dict.items():
        print(mosaic)
        print("Mean score: " + str(np.mean(scores)))
        print("90th percentile score: " + str(np.percentile(scores, 90)))
        print(len(scores))
        print("------------------------")
    

if __name__ == "__main__":
    
    IMAGE_SIZE, IMAGE_CHANNELS = 300, 3
    TOTAL_CLASSES = 14

    model_path = ''
    training_dir = ''

    output_model = mahalanobis_model(model_path)

    val_batch, val_steps = validation_batch_steps(training_dir)

    train_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function = cnn_preprocessing,
        data_format = "channels_last").flow_from_directory(directory = training_dir,
        target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical',
        batch_size = val_batch, shuffle = False)

    mahalanobis_output = output_model.predict_generator(train_generator, steps= val_steps, verbose=1) 

    maha_class = Class_conditional_dicts(layer_list = layer_outputs)
    maha_class.populate_mu_sigmas(mahalanobis_output, train_generator)
    
    mosaic_dict = directory_iterator("")
    export_mahalanobis_scores(mosaic_dict, layer_outputs, metric="mean")

    mosaic = import_raw_dicom("")
    patches = patch_generator(mosaic)
    inference_mah = feedforward(patches, model=output_model)
    mosaic_dict = mahalanobis_score_ensemble(maha_class, inference_mah)