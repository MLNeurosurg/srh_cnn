
'''
Model performance script
1) Accuracy
2) ROCs
3) Error analysis
4) Mosaic level predictions/evaluations

'''

from evaluation import *
import os
import numpy as np

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
#from keras.applications.densenet import DenseNet121

######## Import model
#base_model = DenseNet121(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
base_model = InceptionResNetV2(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(.5)(x)  
x = Dense(20, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dense(total_classes, kernel_initializer='he_normal')(x)
predictions = Activation('softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#model.save_weights('my_model_weights.h5')
#model.load_weights('/home/todd/Desktop/weights.03-0.87.hdf5')
#model = load_model("/home/todd/Desktop/densenet_NIOINV_trainacc867valacc814.hdf5")


# Distribute model across GPUs
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.load_weights("/home/todd/Desktop/Models/Final_Resnet_weights.03-0.86PAPERMODEL.hdf5")

def cnn_predictions(model, validation_generator):
    cnn_predictions = model.predict_generator(validation_generator, steps = val_steps, verbose = True)
    return cnn_predictions

'''
WITH NORMAL
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
 'schwannoma':12,
 'whitematter': 13}

WITHOUT NORMAL and Gliosis
{'ependymoma': 0,
 'glioblastoma': 1,
 'lowgradeglioma': 2,
 'lymphoma': 3,
 'medulloblastoma': 4,
 'meningioma': 5,
 'metastasis': 6,
 'nondiagnostic': 7,
 'pilocyticastrocytoma': 8,
 'pituitaryadenoma': 9,
 'schwannoma: 10}

'''

# Specify directory with images for prediction
validation_dir = '/home/todd/Desktop/CNN_Images/columbia_trial_tiles'

# Define batch size and number of steps
val_batch, val_steps = validation_batch_steps(validation_dir)

# instantiate validation generator
validation_generator = ImageDataGenerator(
    samplewise_center=False,
    samplewise_std_normalization = False,
    horizontal_flip=False,
    vertical_flip=False,
    preprocessing_function=nio_preprocessing_function,
    data_format = "channels_last").flow_from_directory(directory = validation_dir,
    target_size = (img_rows, img_cols), color_mode = 'rgb', classes = None, class_mode = 'categorical',
    batch_size=val_batch, shuffle = False)

# Model predictions
preds = cnn_predictions(parallel_model, validation_generator)

# instantiate model object
model_object = TrainedCnnModel(parallel_model, validation_generator, preds)

#
# evaluate model performance
bar = model_object.mosaic_softmax_all_normal_filter()
baz = model_object.patient_softmax_all_normal_filter()

## multiclass confusion matrices
#tile_level_multiclass_confusion_matrix(model_object, normalize=True)
#mosaic_level_multiclass_confusion_matrix(model_object, normalize=True)
#patient_level_multiclass_confusion_matrix(model_object, normalize=True)
#

test = model_object.IOU_inference_classes("NIO158-3265_43")

mosaic = "NIO088-6438_3"
single_model_heatmap(model_object, mosaic, 4, cmap='Spectral_r', heatmap_type = "ground_truth")

IOU_inference_list = []
for mosaic in model_object.mosaic_list():
    print(mosaic)
    try:
        IOU_inference_list.append(model_object.IOU_inference_classes(mosaic))
    except:
        continue
    
iou_inference_dict = {}
for i in IOU_inference_list:
    for key, value in i.items():
        iou_inference_dict[key] = value

import pandas as pd
iou_df = pd.DataFrame((iou_inference_dict))        
iou_df.to_excel("columbia_inference_iou.xlsx")

tumor_mean = list(iou_df.iloc[0, :])
tumor_mean = [val for val in tumor_mean if val >= 0.3]
np.mean(tumor_mean)

heatmap = model_object.diagnosis_heatmap(mosaic, 8, heatmap_type="ground_truth")
plt.imshow(heatmap, cmap='Spectral_r', vmin=0,vmax=1)
matplotlib.pyplot.imsave("NIO088_lymphoma.png", heatmap, cmap='Spectral_r', vmin=0,vmax=1)



from PIL import Image
img = Image.fromarray((heatmap * 255).astype(np.uint8))
img.save("NIO088_non_gs.png")



