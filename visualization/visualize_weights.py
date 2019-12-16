
"""
Evaluate a specific convolutional layer weights for trained CNN
"""

from imageio import imread
from skimage.transform import resize
from scipy.misc import imresize
from keras.models import load_model
from keras.applications import InceptionResNetV2

def rescale(image):
	image -= image.mean()
	image /= image.std()
	image *= 64
	image += 128
	return image.clip(min=0, max=255).astype("uint8")

def plot_layer_filters(model, layer = 1, interpolate = True):
    # Evaluate layer of convolutions for trained CNN
    cnn_first_layer = model.get_layer(index = layer).get_weights()
    cnn_first_layer = np.asanyarray(cnn_first_layer)[0, : , :, :]

    for i in range(32):
        plt.subplot(4,8,i + 1)
        plt.axis("off")
        
        if interpolate:
            plt.imshow(imresize(rescale(cnn_first_layer[:,:,:,i]), (25, 25), interp="bicubic"))
        else:
            plt.imshow(rescale(cnn_first_layer[:,:,:,i]))

    plt.show()

if __name__ == "__main__":

    # Load models of interest
    model = load_model("")
    model_imagenet = InceptionResNetV2(weights="imagenet")

    plot_layer_filters(model, interpolate=False)
