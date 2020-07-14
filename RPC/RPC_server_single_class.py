'''
  RPC Service
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
from xmlrpc.server import SimpleXMLRPCServer
import keras
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
from LIBS.ImgPreprocess.my_preprocess import do_preprocess
from LIBS.ImgPreprocess.my_image_helper import my_gen_img_tensor
import my_config

PREPROCESS_IMG_SIZE = my_config.preprocess_img_size

def predict_softmax(img1, preproess=False):
    if preproess:
        img1 = do_preprocess(img1, crop_size=PREPROCESS_IMG_SIZE)

    prob_np = []
    prob = []
    pred = []  # only one class, multi_models multi-class multi-label has multiple classes

    for dict1 in dicts_models:
        #real time image aug
        img_tensor = my_gen_img_tensor(img1,image_shape=dict1['input_shape'])
        prob1 = dict1['model'].predict_on_batch(img_tensor)
        prob1 = np.mean(prob1, axis=0)  # batch mean, test time img aug
        pred1 = prob1.argmax(axis=-1)

        prob_np.append(prob1)  #  numpy  weight avg prob_total

        prob.append(prob1.tolist())    #担心XMLRPC numpy
        pred.append(int(pred1))   # numpy int64, int  XMLRPC

    list_weights = []  # the prediction weights of models
    for dict1 in dicts_models:
        list_weights.append(dict1['model_weight'])

    prob_total = np.average(prob_np, axis=0, weights=list_weights)
    pred_total = prob_total.argmax(axis=-1)

    prob_total = prob_total.tolist()  #RPC Service can not pass numpy variable
    pred_total = int(pred_total)     # 'numpy.int64'  XMLRPC

    # correct_model_no is used for choosing which model to generate CAM
    # on extreme condition: average softmax prediction class is not in every model's prediction class
    correct_model_no = 0
    for i, pred1 in enumerate(pred):
        if pred1 == pred_total:
            correct_model_no = i    #start from 0
            break

    return prob, pred, prob_total, pred_total, correct_model_no

#command parameters: predict class type no and port number
if len(sys.argv) == 3:  # sys.argv[0]  exe file itself
    reference_class = str(sys.argv[1])
    port = int(sys.argv[2])
else:
    reference_class = '1'  # left_right_eye
    port = 5001


#region define models
model_dir = my_config.dir_deploy_models
dicts_models = []

# image quality
if reference_class == '0':
    model_file1 = os.path.join(model_dir,
         'image_quality/2020_2_24',  'Xception-010-0.957.hdf5')
    dict_model1 = {'model_file': model_file1, 'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)

# left right eye
if reference_class == '1':
    model_file1 = os.path.join(model_dir,
         'left_right_eye',  'InceptionV3-007-1.000.hdf5')
    dict_model1 = {'model_file': model_file1, 'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)

#stage
if reference_class == '2':
    model_file1 = os.path.join(model_dir, 'STAGE/2020_3_7', 'Xception-010-0.981.hdf5')
    dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)
    # model_file1 = os.path.join(DIR_MODELS, 'STAGE/2020_3_7',  'Xception-010-0.981.hdf5')
    # dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    # dicts_models.append(dict_model1)
    # model_file1 = os.path.join(DIR_MODELS, 'STAGE/2020_3_7',  'InceptionV3-010-0.979.hdf5')
    # dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    # dicts_models.append(dict_model1)

# hemorrhage
if reference_class == '3':
    model_file1 = os.path.join(model_dir,
         'hemorrhage/2020_3_7',  'Xception-005-0.989.hdf5')
    dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)
    # model_file1 = os.path.join(DIR_MODELS,
    #      'hemorrhage/2020_3_7',  'InceptionResnetV2-007-0.993.hdf5')
    # dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    # dicts_models.append(dict_model1)
    # model_file1 = os.path.join(DIR_MODELS,
    #      'hemorrhage/2020_3_7',  'InceptionV3-008-0.991.hdf5')
    # dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    # dicts_models.append(dict_model1)

# posterior
if reference_class == '4':
    model_file1 = os.path.join(model_dir, 'posterior/2020_2_24', 'mobilenet_v2-010-0.970.hdf5')
    dict_model1 = {'model_file': model_file1,  'input_shape': (224, 224, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)
    # model_file1 = os.path.join(DIR_MODELS, 'posterior/2020_2_24', 'InceptionV3-005-0.978.hdf5')
    # dict_model1 = {'model_file': model_file1,  'input_shape': (224, 224, 3), 'model_weight': 1}
    # dicts_models.append(dict_model1)


# plus two stages
if reference_class == '6':
    model_file1 = os.path.join(model_dir,
         'plus_two_stages/2020_4_28',  'Xception-005-0.967.hdf5')
    dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)
    model_file1 = os.path.join(model_dir,
         'plus_two_stages/2020_4_28',  'InceptionResnetV2-008-0.973.hdf5')
    dict_model1 = {'model_file': model_file1,  'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)
#endregion

#load models
for dict1 in dicts_models:
    model_file = dict1['model_file']
    print('%s load start!' % (model_file))
    # ValueError: Unknown activation function:relu6  MobileNet V2
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        dict1['model'] = keras.models.load_model(model_file, compile=False)

    if 'input_shape' not in dict1:
        if len(dict1['model'].input_shape) == 4: #(batch, height, width, channel)
            dict1['input_shape'] = dict1['model'].input_shape[1:]
        else:
            dict1['input_shape'] = (299, 299, 3)

    print('%s load complte!' % (model_file))



#region test mode
if my_config.debug_mode:
    img_source = '/media/ubuntu/data1/ROP_dataset/Stage/preprocess384/广州妇幼2017-2018/分期病变/1089fd0e-2aeb-4428-997d-02db4b9a3980.10.jpg'

    if os.path.exists(img_source):
        img1 = do_preprocess(img_source, PREPROCESS_IMG_SIZE)

        prob_list, pred_list, prob_total, pred_total, correct_model_no = predict_softmax(img1)
        print(prob_total)
    else:
        print('file:', img_source, ' does not exist.')
#endregion


server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
server.register_function(predict_softmax, "predict_softmax")
server.serve_forever()

