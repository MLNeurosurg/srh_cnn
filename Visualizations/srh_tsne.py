'''
Data Visualization for CNN

'''
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from pandas import DataFrame
from keras import models

# Keras Deep Learning modules
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator # may not need this
# Model and layer import
from keras.utils import multi_gpu_model
from keras.models import Sequential, Model, Input
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D 
# Open-source models
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2


validation_dir = '/home/todd/Desktop/tsne_patches/tsne'

# Image specifications/interpolation
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

mapping = {0: 'ependymoma', 1: 'glioblastoma', 2: 'greymatter', 3: 'lowgradeglioma', 4: 'lymphoma', 5: 'medulloblastoma', 6: 'meningioma', 7: "metastasis", 
           8: "nondiagnostic", 9: "pilocyticastrocytoma", 10: "pituitaryadenoma", 11: "pseudoprogression", 12: "whitematter", 13: "schwannoma"}

def nio_preprocessing_function(image):
	image[:,:,0] -= 102.1
	image[:,:,1] -= 91.0
	image[:,:,2] -= 101.5
	return(image)


def find_pair_factors_for_CNN(x):
    """
    Function to match batch size and iterations for the validation generator
    """
    pairs = []
    for i in range(2, 60):
        test = x/i
        if i * int(test) == x:
            pairs.append((i, int(test)))
    best_pair = pairs[-1]
    return best_pair

def validation_batch_steps(directory):
    counter = 0
    for roots, dirs, files in os.walk(directory):
        for file in files:
            counter += 1
    return find_pair_factors_for_CNN(counter)

val_batch, val_steps = validation_batch_steps(validation_dir)

validation_generator = ImageDataGenerator(
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function = nio_preprocessing_function,
    data_format = "channels_last").flow_from_directory(directory = validation_dir,
    target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical', 
    batch_size = val_batch, shuffle = False)


###############################################################
# 1) TSNE model
######## Import model
#base_model = DenseNet121(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
#base_model = InceptionResNetV2(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
#
## Add a global spatial average pooling layer
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dropout(.5)(x)  
#x = Dense(20, kernel_initializer='he_normal')(x)
#x = BatchNormalization()(x)
#x = Activation("relu")(x)
#x = Dense(total_classes, kernel_initializer='he_normal')(x)
#predictions = Activation('softmax')(x)
#model = Model(inputs=base_model.input, outputs=predictions)
#
## Distribute model across GPUs
#parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.load_weights("/home/todd/Desktop/Models/Final_Resnet_weights.03-0.86PAPERMODEL.hdf5")

model = load_model("/home/todd/Desktop/Models/model_for_activations_Final_Resnet_weights.03-0.86.hdf5")

#preds= model.predict_generator(validation_generator, steps=val_steps, verbose=1)
#cnn_predict_1d = np.argmax(preds, axis = 1)   
## Ground truth generated from the validation generator
#index_validation = validation_generator.classes
## Overall accuracy
#from sklearn.metrics import accuracy_score
#accuracy_score(index_validation, cnn_predict_1d)

# build TSNE model
model_tsne = models.Model(input=model.inputs, outputs=model.layers[-7].output) # indexing into the global average pooling layer, -7, -3

# Pass the images to obtain the 2048 vector output
cnn_representations = model_tsne.predict_generator(validation_generator, steps=val_steps, verbose = 1)

from sklearn.decomposition import PCA
pca = PCA(n_components = 20)
cnn_reps_pca = pca.fit_transform(cnn_representations)

# load TSNE module
from sklearn.manifold import TSNE
tsne = TSNE(learning_rate=200, perplexity=50, verbose=1)
tsne_features = tsne.fit_transform(cnn_reps_pca)
s
xs = tsne_features[:,0]
ys = tsne_features[:,1]
tsne_df = DataFrame(np.column_stack((xs, ys, validation_generator.classes)), columns = ['xs', 'ys','labels'])
# rename the categories 
tsne_df = tsne_df.replace({'labels': mapping})
tsne_df['labels'] = tsne_df['labels'].astype('category')
tsne_df['filenames'] = validation_generator.filenames

# scatterplot of TSNE 2D representation 
import seaborn as sns
sns.lmplot('xs', 'ys', data=tsne_df, hue='labels', fit_reg=False)

tsne_df.to_excel("tsne_df.xlsx")

