'''
Functions and subroutines to evaluate predictions of SRH-CNN model over full mosaics and patients.
This scripts is designed specifically to import patches that have already been generated and saved using an older preprocessing routine. 

A MUCH easier method is to predict starting from raw SRH strips using:
    1) import_srh_dicom
    2) patch_generator
    3) prediction
    4) heatmaps
'''
 
import os
from collections import defaultdict, OrderedDict

# Model evaluation
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Keras Deep Learning modules
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator # may not need this
from keras.utils.np_utils import to_categorical

# Image processing
from PIL import ImageFile
from skimage.filters import gaussian
from skimage.io import imsave

ImageFile.LOAD_TRUNCATED_IMAGES = True
img_rows = 300
img_cols = 300
img_channels = 3
total_classes = 14

class_names = ['ependymoma',
 'glioblastoma',
 'greymatter',
 'lowgradeglioma',
 'lymphoma',
 'medulloblastoma',
 'meningioma',
 'metastasis',
 'nondiagnostic',
 'pilocyticastrocytoma',
 'pituitaryadenoma',
 'pseudoprogression',
 'schwannoma',
 'whitematter']

class_names_without_normal = ['ependymoma',
 'glioblastoma',
 'lowgradeglioma',
 'lymphoma',
 'medulloblastoma',
 'meningioma',
 'metastasis',
 'pilocyticastrocytoma',
 'pituitaryadenoma',
 'schwannoma']

class_dict = {'ependymoma': 0,
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
 'schwannoma':12,
 'whitematter': 13}

# Dictionary that contains the patch-heatmap pixel mappings for local probability pooling, DO NOT CHANGE
tile_index_dict = {
0:[0],
1:[0,1],
2:[0,1,2],
3:[1,2,3],
4:[2,3,4],
5:[3,4,5],
6:[4,5,6],
7:[5,6,7],
8:[6,7],
9:[7],
10:[0,8],
11:[0,1,8,9],
12:[0,1,2,8,9,10],
13:[1,2,3,9,10,11],
14:[2,3,4,10,11,12],
15:[3,4,5,11,12,13],
16:[4,5,6,12,13,14],
17:[5,6,7,13,14,15],
18:[6,7,14,15],
19:[7,15],
20:[0,8,16],
21:[0,1,8,9,16,17],
22:[0,1,2,8,9,10,16,17,18],
23:[1,2,3,9,10,11,17,18,19],
24:[2,3,4,10,11,12,18,19,20],
25:[3,4,5,11,12,13,19,20,21],
26:[4,5,6,12,13,14,20,21,22],
27:[5,6,7,13,14,15,21,22,23],
28:[6,7,14,15,22,23],
29:[7,15,23],
30:[8,16,24],
31:[8,9,16,17,24,25],
32:[8,9,10,16,17,18,24,25,26],
33:[9,10,11,17,18,19,25,26,27],
34:[10,11,12,18,19,20,26,27,28],
35:[11,12,13,19,20,21,27,28,29],
36:[12,13,14,20,21,22,28,29,30],
37:[13,14,15,21,22,23,29,30,31],
38:[14,15,22,23,30,31],
39:[15,23,31],
40:[16,24,32],
41:[16,17,24,25,32,33],
42:[16,17,18,24,25,26,32,33,34],
43:[17,18,19,25,26,27,33,34,35],
44:[18,19,20,26,27,28,34,35,36],
45:[19,20,21,27,28,29,35,36,37],
46:[20,21,22,28,29,30,36,37,38],
47:[21,22,23,29,30,31,37,38,39],
48:[22,23,30,31,38,39],
49:[23,31,39],
50:[24,32,40],
51:[24,25,32,33,40,41],
52:[24,25,26,32,33,34,40,41,42],
53:[25,26,27,33,34,35,41,42,43],
54:[26,27,28,34,35,36,42,43,44],
55:[27,28,29,35,36,37,43,44,45],
56:[28,29,30,36,37,38,44,45,46],
57:[29,30,31,37,38,39,45,46,47],
58:[30,31,38,39,46,47],
59:[31,39,47],
60:[32,40,48],
61:[32,33,40,41,48,49],
62:[32,33,34,40,41,42,48,49,50],
63:[33,34,35,41,42,43,49,50,51],
64:[34,35,36,42,43,44,50,51,52],
65:[35,36,37,43,44,45,51,52,53],
66:[36,37,38,44,45,46,52,53,54],
67:[37,38,39,45,46,47,53,54,55],
68:[38,39,46,47,54,55],
69:[39,47,55],
70:[40,48,56],
71:[40,41,48,49,56,57],
72:[40,41,42,48,49,50,56,57,58],
73:[41,42,43,49,50,51,57,58,59],
74:[42,43,44,50,51,52,58,59,60],
75:[43,44,45,51,52,53,59,60,61],
76:[44,45,46,52,53,54,60,61,62],
77:[45,46,47,53,54,55,61,62,63],
78:[46,47,54,55,62,63],
79:[47,55,63],
80:[48,56],
81:[48,49,56,57],
82:[48,49,50,56,57,58],
83:[49,50,51,57,58,59],
84:[50,51,52,58,59,60],
85:[51,52,53,59,60,61],
86:[52,53,54,60,61,62],
87:[53,54,55,61,62,63],
88:[54,55,62,63],
89:[55,63],
90:[56],
91:[56,57],
92:[56,57,58],
93:[57,58,59],
94:[58,59,60],
95:[59,60,61],
96:[60,61,62],
97:[61,62,63],
98:[62,63],
99:[63]}

def nio_preprocessing_function(image):
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
    for i in range(1, 120): # Must restrict the batch size to avoid exhausting GPU memory
        test = x/i
        if i * int(test) == x:
            pairs.append((i, int(test)))
    print(pairs)
    best_pair = pairs[-1]
    print(best_pair)
    return best_pair

def validation_batch_steps(directory):
    """
    Returns the best pair as a tuple of batch size and steps for the validation generator
    """
    counter = 0
    for roots, dirs, files in os.walk(directory):
        for file in files:
            counter += 1
    print(counter)
    return find_pair_factors_for_CNN(counter)

def filename_splitter(filename):
    """
    Function to split following format: metastasis/NIO107-6538_4_tile_061_14.tif
    """
    tumor, file = filename.split("/")
    inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
    return inv_mrn + "_" + mosaic

def fov_tile_splitter(x):
    """
    Accepts a 4-tuple are argument and returns a 6 tuple that includes NIO case number and tile number

    Function used specifically for heatmap generation.
    """
    filename, array, label, tumor = x
    file, tif = filename.split(".")
    _, _, _, fov, tile = file.split("_")
    return (fov, tile, filename, array, label, tumor)

def tiling_fovs_local_prob(mosaic_list, tumortype_int):
    """
    Function that accepts a list of patches from a single mosaics and generates a prediction heatmap for a specified tumor class
    """
    single_mosaic_list = sorted(mosaic_list, key = lambda x: (int(x[0]), int(x[1]))) # sort predictions by case and tile number

    tile_num = len(single_mosaic_list) # number of tiles
    fov_numb = int(tile_num/64) # number of fovs
    mosaic_size = int(np.sqrt(fov_numb)) #1, 3, 4, or 5

    keys = list(range(1, fov_numb + 1)) # naming convention for image indexing for the FOV starts at 1
    single_mosaic_dict = {}
    for key in keys:
        if key not in single_mosaic_dict.keys():
            single_mosaic_dict[key] = []

    for key, val in single_mosaic_dict.items():
        for tuples in single_mosaic_list:
            if int(tuples[0]) == int(key):
                single_mosaic_dict[key].append(tuples)
                
    '''
    Block that pools the neighboring patch probabilities and generates a average propopility for each heatmap pixel
    '''
    long_prob_list = []
    for fovs, tiles_from_fov in single_mosaic_dict.items():

        new_probs = [] # list of new probabilities from operation below. A 100 length list of K (number of output classes) element arrays.
        hundred_array = np.arange(100) # This array is only used for indexing; we are generating a new array from 64 > 100 predictions

        for new_tile in hundred_array:

            tile_list = tile_index_dict[new_tile] # index into the mosaic-tile mapping dictionary above
            placeholder_matrix = np.zeros((len(tile_list),14))

            for index, tile in enumerate(tile_list):
                for tuples in tiles_from_fov:
                    if int(tile) == int(tuples[1]): # tuples[1] = tile number
                        placeholder_matrix[index,:] = tuples[3]

            mean_probs = placeholder_matrix.mean(axis = 0) 
            new_probs.append(mean_probs/(mean_probs.sum() + 1e-8)) # Recreating the softmax function, divide by the sum of the mean_prob vector

        long_prob_list.extend(new_probs) # Must use extend here to generate a single long list

    prob_list = []
    for val in long_prob_list:
        prob_list.append(val[tumortype_int]) # select only the class you specify 

    tile_list = []
    for i in range(fov_numb):
        tile_list.append(np.asarray(prob_list[(i * 100):(i + 1) * 100]).reshape(10, 10)) # each value in the list is a 10 X 10 matrix

    starts = []
    for i in range(mosaic_size):
        starts.append(i * 10) # number of starting positions

    mosaic_array = np.zeros((mosaic_size * 10, mosaic_size * 10))
    mosaic_array_index = 0
    for y in starts:
        for x in starts:
            mosaic_array[y:y+10, x:x+10] = tile_list[mosaic_array_index]
            mosaic_array_index += 1

    return gaussian(mosaic_array, 1)

def tiling_fovs_groundtruth(mosaic_list, tumortype_int):
    
    """
    Function that accepts a list of patches from a single mosaics and generates the ground truth labeled mosaic
    """

    single_mosaic_list = sorted(mosaic_list, key = lambda x: (int(x[0]), int(x[1])))

    tile_num = len(single_mosaic_list) # number of tiles
    fov_numb = int(tile_num/64) # number of fovs
    mosaic_size = int(np.sqrt(fov_numb)) #1, 3, 4, or 5
    
    keys = list(range(1, fov_numb + 1)) # Index for the FOV starts at 1
    single_mosaic_dict = {}
    for key in keys:
        if key not in single_mosaic_dict.keys():
            single_mosaic_dict[key] = []

    for key, val in single_mosaic_dict.items():
        for tuples in single_mosaic_list:
            if int(tuples[0]) == int(key):
                single_mosaic_dict[key].append(tuples) # populate the dictionary with tuples from the list: eg {2: (fov, tile, filename, array, label, tumor)}

    '''
    Block that pools the neighboring patch probabilities and generates a averaged softmax for each heatmap pixel
    '''
    long_prob_list = []
    for fovs, tiles_from_fov in single_mosaic_dict.items():

        new_probs = [] # list of new probabilities from operation below. A 100 length list of K (number of output classes) element arrays.
        hundred_array = np.arange(100) # This array is only used for indexing and remind that we are generating a new array from 64 > 100 predictions

        for new_tile in hundred_array:

            tile_list = tile_index_dict[new_tile] # index into the mosaic-tile mapping dictionary above
            placeholder_matrix = np.zeros((len(tile_list),14))

            for index, tile in enumerate(tile_list):
                for tuples in tiles_from_fov:
                    if int(tile) == int(tuples[1]): # selection only the probabilities from the tiles we need according to the tile-tile mapping
                        ground_truth_array = to_categorical(tuples[4], num_classes=14) # generate 
                        placeholder_matrix[index,:] = ground_truth_array

            mean_probs = placeholder_matrix.mean(axis = 0)
            new_probs.append(mean_probs/mean_probs.sum()) # Recreating the softmax function, divide by the sum of the mean_prob vector

        long_prob_list.extend(new_probs) # Must use extend here to generate a single long list

    prob_list = []
    for val in long_prob_list:
        prob_list.append(val[tumortype_int])

    tile_list = []
    for i in range(fov_numb):
        tile_list.append(np.asarray(prob_list[(i * 100):(i + 1) * 100]).reshape(10, 10)) # each value in the list is a 10 X 10 matrix

    starts = []
    for i in range(mosaic_size):
        starts.append(i * 10) # number of starting positions

    mosaic_array = np.zeros((mosaic_size * 10, mosaic_size * 10))
    mosaic_array_index = 0
    for y in starts:
        for x in starts:
            mosaic_array[y:y+10, x:x+10] = tile_list[mosaic_array_index]
            mosaic_array_index += 1
    return gaussian(mosaic_array, 1)

def intersection_over_union(prediction_heatmap, ground_truth):
    
    prediction_heatmap[prediction_heatmap >= 0.5] = True
    prediction_heatmap[prediction_heatmap < 0.5] = False
    prediction_heatmap = prediction_heatmap.astype(bool)
    
    ground_truth[ground_truth >= 0.5] = True
    ground_truth[ground_truth < 0.5] = False
    ground_truth = ground_truth.astype(bool)

    if ground_truth.sum() == 0:
        return 0
    if prediction_heatmap.sum() == 0:
        return 0
    else:
        # calculate the intersection and union
        intersection = prediction_heatmap*ground_truth # Logical AND
        union = prediction_heatmap + ground_truth # Logical OR
        return intersection.sum()/float(union.sum())

##### Main model object
class TrainedCnnModel():

    def __init__(self, model, validation_generator, cnn_predictions):
        self.model = model
        self.validation_generator = validation_generator
        self.predictions_onehot = cnn_predictions
        self.labels = validation_generator.classes
        self.prediction_vector = np.argmax(cnn_predictions, axis = 1)
        self.class_indices = validation_generator.class_indices
        self.filenames = validation_generator.filenames
        self.filenames_pred_zipped = sorted(zip(self.filenames, self.predictions_onehot, self.labels))

    def mosaic_list(self):
        mosaic_list = []
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_list.append(inv_mrn + "_" + mosaic)
        return sorted(list(set(mosaic_list)))

    def patient_list(self):
        patient_list = []
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            patient_list.append(inv_mrn)
        return sorted(list(set(patient_list)))

    def overall_accuracy(self):
        return accuracy_score(self.labels, self.prediction_vector)

    def filename_prediction(self):
        filenames_pred_zipped = sorted(zip(self.filenames, self.prediction_vector))  # need to zip this
        classnames = list(self.class_indices.keys())

        # First generates a dictionary with keys being tumor classes and values softmax predictions
        file_dict = {}
        for name in classnames:
            if name not in file_dict.keys():
                file_dict[name] = []
            for tuple in filenames_pred_zipped:
                if tuple[0].startswith(name):
                    file_dict[name].append(tuple)
        return file_dict

    def tumor_class_accuracy(self):
        filenames_pred_zipped = dict(sorted(zip(self.filenames, self.prediction_vector))) # need to zip this
        classnames = list(self.class_indices.keys())

        # First generates a dictionary with keys being tumor classes and values softmax predictions
        file_dict = {}
        for name in classnames:
            if name not in file_dict.keys():
                file_dict[name] = []
            for file, pred in filenames_pred_zipped.items():
                if file.startswith(name):
                    file_dict[name].append(pred)

        # Empty dictionary with correct keys
        accuracy_dict = {}
        for key in file_dict.keys():
            if key not in accuracy_dict.keys():
                accuracy_dict[key] = []

        # One hot for each tumor class
        class_indices = self.class_indices
        for key in accuracy_dict.keys():
            for value in file_dict[key]: # This is a list of the softmax predictions generated for each tumor class
                if int(value) == int(class_indices[key]):
                    accuracy_dict[key].append(1)
                else:
                    accuracy_dict[key].append(0)

        # Final dictionary with accuracy for each tumor class
        final_dict = {}
        for key in accuracy_dict.keys():
            try:
                final_dict[key] = sum(accuracy_dict[key])/len(accuracy_dict[key])
            except:
                final_dict[key] = 0
        return final_dict

    def mean_class_accuracy(self):
        acc_list = [x for x in list(self.tumor_class_accuracy().values()) if str(x) != 'nan']
        return sum(acc_list)/len(acc_list)

    def error_files(self):

        filenames_pred_zipped = sorted(zip(self.filenames, self.prediction_vector)) # need to zip this
        ground_truth = validation_generator.classes
        classnames = list(self.validation_generator.class_indices.keys())

        error_list = []
        for index, value in enumerate(filenames_pred_zipped):
            if value[1] != ground_truth[index]:
                error_list.append(value)

        error_dict = {}
        for name in classnames:
            if name not in error_dict.keys():
                error_dict[name] = []
            for file in error_list:
                if file[0].startswith(name):
                    error_dict[name].append(file)

        return error_dict

    def mosaic_accuracies_with_nondiag(self):
        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_dict[inv_mrn + "_" + mosaic].append((file, tile_softmax, ground_truth))

        accuracy_dict = {}
        for mosaics, specs in mosaic_dict.items():
            correct_labels = 0
            num_tiles = len(specs)
            for file, tile_softmax, ground_truth in specs:
                if np.argmax(tile_softmax) == ground_truth: # compare the predicted and ground truth
                    correct_labels += 1
            try:
                accuracy_dict[mosaics] = correct_labels/num_tiles # average number of tiles correct INCLUDING nondiagnostic tiles
            except ZeroDivisionError:
                accuracy_dict[mosaics] = "No diagnostic files"
            
        return accuracy_dict

    def mosaic_accuracies_wo_nondiag(self):
        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_dict[inv_mrn + "_" + mosaic].append((file, tile_softmax, ground_truth))

        accuracy_dict = {}
        for mosaics, specs in mosaic_dict.items():
            correct_labels = 0
            diagnostic_tiles = 0
            for file, tile_softmax, ground_truth in specs:
                if (np.argmax(tile_softmax) != 8):
                    diagnostic_tiles += 1
                    if np.argmax(tile_softmax) == ground_truth: # compare the predicted and ground truth
                        correct_labels += 1
            try:
                accuracy_dict[mosaics] = correct_labels/diagnostic_tiles # average number of tiles correct excluding nondiagnostic tiles
            except ZeroDivisionError:
                accuracy_dict[mosaics] = "No diagnostic files"
        
        return accuracy_dict

    def mosaic_softmax(self, mosaic_number):

        accum_array = np.zeros(14, dtype = float)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            if ((mosaic_number) == (inv_mrn + "_" + mosaic)) and (np.argmax(tile_softmax) != 8): # removing tiles that inference nondiagnostic
                accum_array += tile_softmax

        return accum_array/accum_array.sum()

    def mosaic_softmax_normal_filter(self, mosaic_number):
        
        accum_array = np.zeros(14, dtype = float)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            if ((mosaic_number) == (inv_mrn + "_" + mosaic)) and (np.argmax(tile_softmax) != 8):
                accum_array += tile_softmax

        mosaic_softmax = accum_array/accum_array.sum()
        
        if (mosaic_softmax[2] + mosaic_softmax[11] + mosaic_softmax[13]) > 0.9: # threshold the softmax: if greater than 90% normal, include normal in classification
            return mosaic_softmax
        else:
            mosaic_softmax_wo_normal = np.delete(mosaic_softmax, [2, 8, 11, 13])
            return mosaic_softmax_wo_normal/mosaic_softmax_wo_normal.sum()

    def mosaic_softmax_all(self):
        
        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_dict[inv_mrn + "_" + mosaic].append((file, tile_softmax, ground_truth))

        mosaic_softmax_dict = {}
        for mosaic, specs in mosaic_dict.items():
            accum_array = np.zeros(14, dtype = float)
            ground_truth_array = np.zeros(14, dtype=float)
            for file, tile_softmax, ground_truth in specs:
                if (np.argmax(tile_softmax) != 8):
                    accum_array += tile_softmax # tile_softmax
                    ground_truth_array[ground_truth] += 1
            
            mosaic_softmax = accum_array/accum_array.sum()
            ground_truth_softmax = ground_truth_array/ground_truth_array.sum()
            pred_argmax = np.argmax(mosaic_softmax)
            ground_argmax = np.argmax(ground_truth_softmax)
            
            mosaic_softmax_dict[mosaic] = (np.round(mosaic_softmax, decimals = 3), class_names[ground_argmax], pred_argmax == ground_argmax, class_names[pred_argmax])

        return mosaic_softmax_dict
    
    def mosaic_softmax_all_normal_filter(self):
        
        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_dict[inv_mrn + "_" + mosaic].append((file, tile_softmax, ground_truth))

        mosaic_softmax_dict = {}
        for mosaic, specs in mosaic_dict.items():
            accum_array = np.zeros(14, dtype = float)
            ground_truth_array = np.zeros(14, dtype=float)
            for file, tile_softmax, ground_truth in specs:
                if (np.argmax(tile_softmax) != 8): # groundtruth
                    accum_array += tile_softmax # tile_softmax
                    ground_truth_array[ground_truth] += 1
            
            mosaic_softmax = accum_array/accum_array.sum()
            ground_truth_softmax = ground_truth_array/ground_truth_array.sum()
            
            if (mosaic_softmax[2] + mosaic_softmax[11] + mosaic_softmax[13]) > 0.9:
                pred_argmax = np.argmax(mosaic_softmax)
                ground_argmax = np.argmax(ground_truth_softmax)
                mosaic_softmax_dict[mosaic] = (np.round(mosaic_softmax, decimals = 3), class_names[ground_argmax], pred_argmax == ground_argmax, class_names[pred_argmax])
                
            else:
                mosaic_softmax_wo_normal = np.delete(mosaic_softmax, [2, 8, 11, 13])
                ground_truth_wo_normal = np.delete(ground_truth_softmax, [2, 8, 11, 13])
                
                pred_argmax = np.argmax(mosaic_softmax_wo_normal)
                ground_argmax = np.argmax(ground_truth_wo_normal)
                
                renomalized_softmax = mosaic_softmax_wo_normal/mosaic_softmax_wo_normal.sum()
                mosaic_softmax_dict[mosaic] = (np.round(renomalized_softmax, decimals = 3), class_names_without_normal[ground_argmax], pred_argmax == ground_argmax, class_names_without_normal[pred_argmax])
    
        return mosaic_softmax_dict
    
    
    def patient_softmax_all(self):

        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_dict[inv_mrn].append((file, tile_softmax, ground_truth))

        mosaic_softmax_dict = {}
        for patient, specs in mosaic_dict.items():
            accum_array = np.zeros(14, dtype = float)
            ground_truth_array = np.zeros(14, dtype=float)
            for file, tile_softmax, ground_truth in specs:
                if (np.argmax(tile_softmax) != 8): # groundtruth
                    accum_array += tile_softmax # tile_softmax
                    ground_truth_array[ground_truth] += 1
            
            pred_argmax = np.argmax(accum_array/accum_array.sum())
            ground_argmax = np.argmax(ground_truth_array/ground_truth_array.sum())
            mosaic_softmax_dict[patient] = (np.round(accum_array/accum_array.sum(), decimals = 3), class_names[ground_argmax], pred_argmax == ground_argmax, class_names[pred_argmax])

        return mosaic_softmax_dict
    
    def patient_softmax_all_normal_filter(self):
        
        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            mosaic_dict[inv_mrn].append((file, tile_softmax, ground_truth))

        mosaic_softmax_dict = {}
        for patient, specs in mosaic_dict.items():
            accum_array = np.zeros(14, dtype =float)
            ground_truth_softmax = np.zeros(14, dtype=float)
            for file, tile_softmax, ground_truth in specs:
                if (np.argmax(tile_softmax) != 8): # groundtruth
                    accum_array += tile_softmax # tile_softmax
                    ground_truth_softmax[ground_truth] += 1
            
            mosaic_softmax = accum_array/accum_array.sum()
            ground_truth_softmax = ground_truth_softmax/ground_truth_softmax.sum()
            
            if (mosaic_softmax[2] + mosaic_softmax[11] + mosaic_softmax[13]) > 0.9:
                pred_argmax = np.argmax(mosaic_softmax)
                ground_argmax = np.argmax(ground_truth_softmax)
                mosaic_softmax_dict[patient] = (np.round(mosaic_softmax, decimals = 3), class_names[ground_argmax], pred_argmax == ground_argmax, class_names[pred_argmax])
                
            else:
                mosaic_softmax_wo_normal = np.delete(mosaic_softmax, [2, 8, 11, 13])
                ground_truth_wo_normal = np.delete(ground_truth_softmax, [2, 8, 11, 13])
                
                pred_argmax = np.argmax(mosaic_softmax_wo_normal)
                ground_argmax = np.argmax(ground_truth_wo_normal)
                
                renomalized_softmax = mosaic_softmax_wo_normal/mosaic_softmax_wo_normal.sum()
                mosaic_softmax_dict[patient] = (np.round(renomalized_softmax, decimals = 3), class_names_without_normal[ground_argmax], pred_argmax == ground_argmax, class_names_without_normal[pred_argmax])
                
        correct = 0
        for patient, tuples in mosaic_softmax_dict.items():
            if tuples[1] == tuples[3]:
                correct += 1
        print("Patient-level accuracy: "  + str(correct/len(mosaic_softmax_dict)))
    
        return mosaic_softmax_dict

    def mosaic_errors(self):

        mosaic_softmax_dict = self.mosaic_softmax_all()
        error_dict = {}
        for mosaic, tuples in mosaic_softmax_dict.items():
            if not tuples[2]:
                error_dict[mosaic] = tuples
        return error_dict

    def patient_errors(self):

        patient_softmax_dict = self.patient_softmax_all()
        error_dict = {}
        for patient, tuples in patient_softmax_dict.items():
            if not tuples[2]:
                error_dict[patient] = tuples
        return error_dict
    
    
    def diagnosis_heatmap(self, mosaic_number, tumorclass, heatmap_type = "predictions"):

        mosaic_dict = defaultdict(list)
        for tumorfile, tile_softmax, ground_truth in self.filenames_pred_zipped:
            tumor, file = tumorfile.split("/")
            inv_mrn, mosaic, _ = file.split("_", maxsplit=2)
            if ((mosaic_number) == (inv_mrn + "_" + mosaic)):
                mosaic_dict[inv_mrn + "_" + mosaic].append((file, tile_softmax, ground_truth, tumor))
            
        single_mosaic_list = mosaic_dict[mosaic_number]
        for index, tuples in enumerate(single_mosaic_list):
            single_mosaic_list[index] = fov_tile_splitter(tuples) 
            
        if heatmap_type == "predictions":
            return tiling_fovs_local_prob(single_mosaic_list, tumorclass)
        if heatmap_type == "ground_truth":
            return tiling_fovs_groundtruth(single_mosaic_list, tumorclass)

    def intersection_over_union(self, mosaic_number):
        
        IOU_list = [] # initialize a IOU list
        for tumor_key in class_dict.values():
        
            prediction_heatmap = self.diagnosis_heatmap(mosaic_number, tumor_key)
            ground_truth = self.diagnosis_heatmap(mosaic_number, tumor_key, heatmap_type = 'ground_truth')
            
            IOU_list.append(intersection_over_union(prediction_heatmap, ground_truth))
            
        return IOU_list
    
    def IOU_inference_classes(self, mosaic_number):
        
        IOU_list = defaultdict(list) # initialize a dictionary
        
        mosaic_shape = self.diagnosis_heatmap(mosaic_number, 0).shape # find the shape of mosaic
        
        # tumor IOU
        tumor_pred = np.zeros((mosaic_shape[0], mosaic_shape[1]))
        tumor_ground = np.copy(tumor_pred)
        for i in (0,1,3,4,5,6,7,9,10,12):
            tumor_pred += self.diagnosis_heatmap(mosaic_number, i)
            tumor_ground += self.diagnosis_heatmap(mosaic_number, i, 'ground_truth')
        
        IOU_list[mosaic_number].append(intersection_over_union(tumor_pred, tumor_ground))
            
        # nontumor IOU
        nontumor_pred = np.zeros((mosaic_shape[0], mosaic_shape[1]))
        nontumor_ground = np.copy(nontumor_pred)
        for i in (2, 11, 13):
            nontumor_pred += self.diagnosis_heatmap(mosaic_number, i)
            nontumor_ground += self.diagnosis_heatmap(mosaic_number, i, 'ground_truth')
            
        IOU_list[mosaic_number].append(intersection_over_union(nontumor_pred, nontumor_ground))
        
        # nondiagnostic IOU
        nondiag_pred = self.diagnosis_heatmap(mosaic_number, 8)
        nondiag_ground = self.diagnosis_heatmap(mosaic_number, 8, 'ground_truth')
            
        IOU_list[mosaic_number].append(intersection_over_union(nondiag_pred, nondiag_ground))
        
        return dict(IOU_list)
    
def inference_heatmap(model_object, mosaic_name, inference_node = "tumor", heatmap_type="predictions"):

    if inference_node == "tumor":
        child_nodes = (0,1,3,4,5,6,7,9,10,12)
    if inference_node == "normal":
        child_nodes = (2, 11, 13)
    
    array = np.zeros((60,60))
    for i in child_nodes:
        array += model_object.diagnosis_heatmap(mosaic_name, i, heatmap_type=heatmap_type)
    
    return array
    
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.grid(False, which='major')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def tile_level_multiclass_confusion_matrix(Model_object, normalize = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Below is another option if needed:
    from pandas_ml import ConfusionMatrix
    cm = ConfusionMatrix(index_validation, cnn_predict_1d, labels = class_names)
    cm.plot(cmap = 'gist_heat_r', axis = 'off')

    OR

    df_conf = DataFrame(cm_norm, index=class_names, columns = class_names)
    heatmap(df_conf, cmap='gist_heat_r')
    """

    if normalize:
        cm = confusion_matrix(Model_object.labels, Model_object.prediction_vector)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm[np.isnan(cm_norm)] = 0
        print(accuracy_score(Model_object.labels, Model_object.prediction_vector))

        # Plot normalized multiclass confusion matrix
        plot_confusion_matrix(cm_norm, classes=class_names, title='Multiclass diagnostic confusion matrix: overall accuracy ' + 
                              str(np.round(accuracy_score(Model_object.labels, Model_object.prediction_vector), decimals = 3)))
        return cm_norm

    if not normalize:
        cm = confusion_matrix(Model_object.labels, Model_object.prediction_vector)
        # Plot non-normalized mutliclass confusion matrix
        plot_confusion_matrix(cm, classes=class_names, title='Multiclass diagnostic confusion matrix')
        return cm

def mosaic_level_multiclass_confusion_matrix(model_object, filter_normal = True, normalize = True):
    
    if filter_normal:
        mosaic_dict = model_object.mosaic_softmax_all_normal_filter()
    else: 
        mosaic_dict = model_object.mosaic_softmax_all()
        
    preds = []
    ground_truth = []
    for mosaic, tuples in mosaic_dict.items():
        preds.append(class_dict[tuples[3]])  # argmax of prediction softmax
        ground_truth.append(class_dict[tuples[1]])  # label
    correct = np.count_nonzero(np.asanyarray(preds) == np.asanyarray(ground_truth))
    print(correct/len(preds))

    included_tumors = set(preds) | set(ground_truth)
    included_classes = []
    for i in included_tumors:
        included_classes.append(class_names[i])
    print(included_classes)

    if normalize:
        cm = confusion_matrix(ground_truth, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm[np.isnan(cm_norm)] = 0
        cm_norm[np.isinf(cm_norm)] = 0

        # Plot normalized multiclass confusion matrix
        plot_confusion_matrix(cm_norm, classes=included_classes, title='Multiclass diagnostic confusion matrix: overall accuracy ' + 
                              str(np.round(correct/len(preds), decimals = 3)))
        return cm_norm

    if not normalize:
        cm = confusion_matrix(ground_truth, preds)
        # Plot unnormalized mutliclass confusion matrix
        plot_confusion_matrix(cm, classes=included_classes, title='Multiclass diagnostic confusion matrix')
        return cm


def patient_level_multiclass_confusion_matrix(model_object, filter_normal = True, normalize = True):
    
    if filter_normal:
        patient_dict = model_object.patient_softmax_all_normal_filter()
    else: 
        patient_dict = model_object.patient_softmax_all()
        
    preds = []
    ground_truth = []
    for mosaic, tuples in patient_dict.items():
        preds.append(class_dict[tuples[3]])  # argmax of prediction softmax
        ground_truth.append(class_dict[tuples[1]])  # label
    correct = np.count_nonzero(np.asanyarray(preds) == np.asanyarray(ground_truth))

    included_tumors = set(preds) | set(ground_truth)
    included_classes = []
    for i in included_tumors:
        included_classes.append(class_names[i])
    print(included_classes)

    if normalize:
        cm = confusion_matrix(ground_truth, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm[np.isnan(cm_norm)] = 0
        cm_norm[np.isinf(cm_norm)] = 0

        # Plot normalized multiclass confusion matrix
        plot_confusion_matrix(cm_norm, classes=included_classes, title='Multiclass diagnostic confusion matrix: overall accuracy ' + 
                              str(np.round(correct/len(preds), decimals = 3)))        
        return cm_norm

    if not normalize:
        cm = confusion_matrix(ground_truth, preds)
        # Plot non-normalized mutliclass confusion matrix
        plot_confusion_matrix(cm, classes=included_classes, title='Multiclass diagnostic confusion matrix')
        return cm

def single_model_heatmap(model_object, mosaic_case, tumortype, heatmap_type = "predictions", cmap='Spectral_r'):
    if heatmap_type == "predictions":
        heatmap = model_object.diagnosis_heatmap(mosaic_case, tumortype)
        
    if heatmap_type == "ground_truth":
        heatmap = model_object.diagnosis_heatmap(mosaic_case, tumortype, heatmap_type = "ground_truth")

    plt.imshow(heatmap, cmap = cmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.title(mosaic_case + ": " + class_names[tumortype])
    plt.colorbar()
    plt.show()

def prediction_goundtruth_heatmaps(model_object, mosaic_case, tumortype, cmap='Spectral_r'):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(model_object.diagnosis_heatmap(mosaic_case, tumortype), cmap = cmap, vmin=0, vmax=1)
    ax1.axis("off")
    ax1.set_title("CNN predictions")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(model_object.diagnosis_heatmap(mosaic_case, tumortype, heatmap_type = "ground_truth"), cmap = cmap, vmin=0, vmax=1)
    ax2.axis('off')
    ax2.set_title("Ground truth")
    iou = model_object.intersection_over_union(mosaic_case, tumortype)
    plt.suptitle(mosaic_case + ": IOU = " + str(np.round(iou, decimals=3)))
    plt.show()


########## ROC analysis
class ROCAnalysis():
    def __init__(self, model, validation_generator, cnn_predictions):
        self.model = model
        self.validation_generator = validation_generator
        self.predictions_onehot = cnn_predictions
        self.labels = validation_generator.classes
        self.prediction_vector = np.argmax(cnn_predictions, axis = 1)
        self.class_indices = validation_generator.class_indices
        self.filenames = validation_generator.filenames

    def nondiagnostic(self):
        ### Diagnostic versus everything else
        '''
        {'ependymoma': 0,
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
        '''
        nondiagnostic_onehot = []
        for i in self.labels:
            if i == 8: # Nondiagnostic
                nondiagnostic_onehot.append(1)
            else:
                nondiagnostic_onehot.append(0)
        nondiagnostic_labels = np.asarray(nondiagnostic_onehot)
        nondiagnostic_predictions = self.predictions_onehot[:,8]
        return (nondiagnostic_labels, nondiagnostic_predictions)

    def normal(self):
        ### White/Grey matter versus everything else
        # Remove nondiagnostic tiles
        trimmed_predictions = self.predictions_onehot[self.labels != 8]
        timmed_labels = self.labels[self.labels != 8]

        normal_onehot = []
        for i in timmed_labels:
            if (i == 2) or (i == 12): # Grey matter or white matter
                normal_onehot.append(1)
            else:
                normal_onehot.append(0)
        normal_labels = np.asarray(normal_onehot)
        normal_predictions = trimmed_predictions[:,(2, 13)].sum(axis=1)
        return (normal_labels, normal_predictions)

    def nonsurgical(self):
        ### Lymphoma and pseudoprogression
        # Remove nondiag (8), greymatter(2), whitematter(12)
        # trimmed_predictions = self.predictions_onehot[np.logical_and.reduce((self.labels != 8, self.labels != 2, self.labels != 12))]
        # trimmed_labels = self.labels[np.logical_and.reduce((self.labels != 8, self.labels != 2, self.labels != 12))]

        trimmed_predictions = self.predictions_onehot[(self.labels != 8) & (self.labels != 2) & (self.labels != 12)]
        trimmed_labels = self.labels[(self.labels != 8) & (self.labels != 2) & (self.labels != 12)]

        nonsurgical_onehot = []
        for i in trimmed_labels:
            if (i == 4): # Lymphoma
                nonsurgical_onehot.append(1)
            else:
                nonsurgical_onehot.append(0)
        nonsurgical_labels = np.asarray(nonsurgical_onehot)
        nonsurgical_predictions = trimmed_predictions[:,4]
        return (nonsurgical_labels, nonsurgical_predictions)

    def glial(self):
        ### Ependymoma, Glioblastoma, LGG, Pilocytic
        # Remove nondiag(8), greymatter(2), whitematter(12), lymphoma (4), pseudoprogression (11)
        # trimmed_predictions = self.predictions_onehot[np.logical_and.reduce((self.labels != 8, self.labels != 2, self.labels != 12, self.labels != 4))]
        # trimmed_labels = self.labels[np.logical_and.reduce((self.labels != 8, self.labels != 2, self.labels != 12, self.labels != 4))]

        trimmed_predictions = self.predictions_onehot[(self.labels != 8) & (self.labels != 2) & (self.labels != 12) & (self.labels != 4) & (self.labels != 11)]
        trimmed_labels = self.labels[(self.labels != 8) & (self.labels != 2) & (self.labels != 12) & (self.labels != 4) & (self.labels != 11)]

        glial_onehot = []
        for i in trimmed_labels:
            if (i == 0) or (i == 1) or (i == 3) or (i == 9):
                glial_onehot.append(1)
            else:
                glial_onehot.append(0)
        glial_labels = np.asarray(glial_onehot)
        glial_predictions = trimmed_predictions[:,(0, 1, 3, 9)].sum(axis=1)
        return (glial_labels, glial_predictions)

    def malignant_glioma(self):
        # FLIP to selecting ONLY appropriate cases
        # trimmed_predictions = self.predictions_onehot[np.logical_or.reduce((self.labels == 0, self.labels == 1, self.labels == 3, self.labels == 9))]
        # trimmed_labels = self.labels[np.logical_or.reduce((self.labels == 0, self.labels == 1, self.labels == 3, self.labels == 9))]

        trimmed_predictions = self.predictions_onehot[(self.labels == 0) | (self.labels == 1) | (self.labels == 3) | (self.labels == 9)]
        trimmed_labels = self.labels[(self.labels == 0) | (self.labels == 1) | (self.labels == 3) | (self.labels == 9)]

        malignant_glial_onehot = []
        for i in trimmed_labels:
            if (i == 1):
                malignant_glial_onehot.append(1)
            else:
                malignant_glial_onehot.append(0)
        malignant_glial_labels = np.asarray(malignant_glial_onehot)
        malignant_glial_predictions = trimmed_predictions[:,1]
        return (malignant_glial_labels, malignant_glial_predictions)

    def lowgradeglioma(self):
        # FLIP to selecting ONLY appropriate cases
        # trimmed_predictions = self.predictions_onehot[np.logical_or.reduce((self.labels == 0, self.labels == 3, self.labels == 9))]
        # trimmed_labels = self.labels[np.logical_or.reduce((self.labels == 0, self.labels == 3, self.labels == 9))]

        trimmed_predictions = self.predictions_onehot[(self.labels == 0) | (self.labels == 3) | (self.labels == 9)]
        trimmed_labels = self.labels[(self.labels == 0) | (self.labels == 3) | (self.labels == 9)]

        lowgradeglioma_onehot = []
        for i in trimmed_labels:
            if (i == 3):
                lowgradeglioma_onehot.append(1)
            else:
                lowgradeglioma_onehot.append(0)
        lowgradeglioma_labels = np.asarray(lowgradeglioma_onehot)
        lowgradeglioma_predictions = trimmed_predictions[:,3]
        return (lowgradeglioma_labels, lowgradeglioma_predictions)


    def roc_plotting(self, decision_point):
        if decision_point == "nondiagnostic":
            labels, predictions = self.nondiagnostic()

        if decision_point == "normal":
            labels, predictions = self.normal()

        if decision_point == "nonsurgical":
            labels, predictions = self.nonsurgical()

        if decision_point == "glial":
            labels, predictions = self.glial()

        if decision_point == "malignant_glioma":
            labels, predictions = self.malignant_glioma()

        if decision_point == "lowgradeglioma":
            labels, predictions = self.lowgradeglioma()

        plt.figure(0).clf()
        plt.title('Receiver Operating Characteristic')
#        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.show()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        fpr, tpr, thresh = roc_curve(labels, predictions)
        auc = roc_auc_score(labels, predictions)
        plt.plot(fpr, tpr, label = "AUC = " + str(round(auc,3)))
        plt.legend(loc=4)


    print("SRH_CNN evaluation script imported.")
    

if __name__ == "__main__":
    pass