'''https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py'''
import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.python.framework import ops


H, W = 299, 299  # Input shape, defined by the model (model.input_shape)

# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model(model):
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = model
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    return cam



if __name__ == '__main__':
    import keras

    model_file = '/tmp3/models_ROP_2019_11_14/Grade/train0/transfer0/InceptionV3-003-0.976.hdf5'
    model1 = keras.models.load_model(model_file, compile=False)

    image_file = '/media/ubuntu/data1/ROP项目/preprocess384/ROP训练图集汇总_20190928/本院201808-201908/分期病变/201901190009_OD201901190013_左眼_20190119100434986.jpg'

    from LIBS.DataPreprocess import my_images_generator
    from LIBS.ImgPreprocess import my_preprocess
    preprocess = False
    image_size = 299
    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                         image_shape=(image_size, image_size, 3))
    else:
        img_source = image_file
        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                                         image_shape=(image_size, image_size, 3))

    probs = model1.predict(img_input)
    class_predict = np.argmax(probs)

    from LIBS.CNN_Models.Utils.my_utils import get_last_conv_layer_name
    layer_name = get_last_conv_layer_name(model1)

    guided_model = build_guided_model(model1)

    preprocessed_input = img_input

    gradcam = grad_cam(model1, preprocessed_input, class_predict, layer_name)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]


    jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    jetcam = (np.float32(jetcam) + np.squeeze(img_input)) / 2
    cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
    cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
    cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))


    print('OK')