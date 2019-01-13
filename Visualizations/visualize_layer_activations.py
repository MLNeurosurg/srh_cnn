

from keras.models import load_model
from keras.models import Model
from imageio import imread
import numpy as np
import sys
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from keras.applications import InceptionResNetV2


# model = InceptionResNetV2(weights = "imagenet")
model = load_model("/media/todd/TODD/model_for_activations_Final_Resnet_weights.03-0.86.hdf5")
model_imagenet = InceptionResNetV2(weights="imagenet")

img = imread("/home/todd/Desktop/NIO105-1058_3_tile_013_6.tif").astype(np.float64)
test = gaussian(img, sigma=2)


img_imagenet = imread("/home/todd/Desktop/columbine-2.png").astype(np.float64)
img_imagenet = gaussian(img_imagenet, sigma=1)
plt.imshow(img_imagenet[:,:,0])
plt.colorbar()

#base_model = InceptionV3(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3)) # must include the input shape to use keras-vis!!!
base_model = InceptionResNetV2(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
#base_model = DenseNet121(weights=None, include_top=False, input_shape = (img_rows, img_cols, 3))
# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(.5)(x)  
x = Dense(20, kernel_initializer='he_normal')(x)
x = BatchNormalization(name='todd_batch_norm')(x)
x = Activation("relu", name='todd_activation')(x)
x = Dense(total_classes, kernel_initializer='he_normal')(x)
predictions = Activation('softmax', name='todd_activation_2')(x)
test_model = Model(inputs=base_model.input, outputs=predictions)


def nio_preprocessing_function(image):
    """
    Channel-wise means calculated over training dataset
    """
    image[:,:,0] -= 102.1
    image[:,:,1] -= 91.0
    image[:,:,2] -= 101.5
    return image

def rescale(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    return image.clip(min = 0, max=255).astype("uint8")

def plot_activations(model, image, activation_layer):
    activations_model = Model(inputs=model.input, outputs = model.layers[activation_layer].output) 
    activations = activations_model.predict(nio_preprocessing_function(image[None,:,:,:]))
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.axis("off")
        plt.imshow(rescale(activations[0,:,:,i]))
    plt.show()

def plot_activations_imagenet(model, image, activation_layer):
    activations_model = Model(inputs=model.input, outputs = model.layers[activation_layer].output) 
    activations = activations_model.predict(image[None,:,:,:] - image.mean())
    for i in range(32):
        plt.subplot(4, 8, i + 1)
        plt.axis("off")
        plt.imshow(rescale(activations[0,:,:,i]))
    plt.show()
    
    


plot_activations_imagenet(model_imagenet, test, activation_layer = 3)

