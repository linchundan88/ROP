
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
GPU_NUM = 2
import keras
import collections
import numpy as np
from LIBS.DLP.my_train_multiclass_helper import train_task_one_step

#region setting train parameters
TRAIN_TYPE = 'Plus_step_two'
DATFILE_TYPE = 'dataset1'
FILENAME_CSV_TRAIN = os.path.join(os.path.abspath('..'),
              'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_train.csv')
FILENAME_CSV_VALID = os.path.join(os.path.abspath('..'),
              'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_valid.csv')
FILENAME_CSV_TEST = os.path.join(os.path.abspath('..'),
              'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_test.csv')
MODEL_SAVE_BASEDIR = os.path.join('/media/ubuntu/data1/tmp5/Plus_step_two_2020_3_14/', TRAIN_TYPE)

from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.96, aug)
IMGAUG_TRAIN_SEQ = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images

    sometimes(iaa.ContrastNormalization((0.9, 1.1))),
    iaa.Sometimes(0.9, iaa.Add((-6, 6))),
    sometimes(iaa.Affine(
        scale=(0.98, 1.02),
        translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
        rotate=(-15, 15),  # rotate by -10 to +10 degrees
    )),
])


'''
3212
0 2560
1 652
381
0 296
1 85
476
0 364
1 112
'''

LABEL_SMOOTHING = 0.1

#positive weight much smaller than pluone stage classificatio
CLASS_WEIGHT = {0: 1., 1: 2.5}

WEIGHT_CLASS_START = np.array([1, 2])
WEIGHT_CLASS_END = np.array([1, 2])
BALANCE_RATIO = 0.93

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64
#endregion

#region training
PRE_TRAIN_TYPE = 'ImageNet'  # fundus Imagenet
TRAIN_TIMES = 3

for i in range(TRAIN_TIMES):
    for MODEL_NAME in ['Xception']:
    # for model_name in ['InceptionResnetV2', 'Xception', 'InceptionV3', 'mobilenet_v2', 'NASNetMobile']:
        MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_BASEDIR, str(i), MODEL_NAME)

        if MODEL_NAME == 'InceptionV3':
            if PRE_TRAIN_TYPE == 'ImageNet':
                model1 = keras.applications.inception_v3.InceptionV3(input_shape=None, include_top=True, weights='imagenet', classes=1000)
            if PRE_TRAIN_TYPE == 'Fundus':
                model1 = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'

            IMAGE_SHAPE = (299, 299, 3)

        if MODEL_NAME == 'InceptionResnetV2':
            if PRE_TRAIN_TYPE == 'ImageNet':
                model1 = keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=None, include_top=True,
                                                                      weights='imagenet', classes=1000)
            if PRE_TRAIN_TYPE == 'Fundus':
                model1 = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/InceptionResNetV2-010-0.958.hdf5'
            IMAGE_SHAPE = (299, 299, 3)

        if MODEL_NAME == 'Xception':
            if PRE_TRAIN_TYPE == 'ImageNet':
                model1 = keras.applications.xception.Xception(input_shape=None, include_top=True,
                                                                      weights='imagenet', classes=1000)
            if PRE_TRAIN_TYPE == 'Fundus':
                model1 = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Xception-008-0.957.hdf5'
            IMAGE_SHAPE = (299, 299, 3)

        if MODEL_NAME == 'mobilenet_v2':
            model1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True,
                                                                     weights='imagenet', classes=1000)
            IMAGE_SHAPE = (224, 224, 3)

        if MODEL_NAME == 'NASNetMobile':
            model1 = keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet',
                                                                input_tensor=None, pooling=None, classes=1000)
            IMAGE_SHAPE = (224, 224, 3)


        EPOCH_FINETUNING = 12

        DICT_LR_FINETUNIN = collections.OrderedDict()
        DICT_LR_FINETUNIN['0'] = 1e-3
        DICT_LR_FINETUNIN['4'] = 3e-4
        DICT_LR_FINETUNIN['5'] = 1e-4
        DICT_LR_FINETUNIN['6'] = 3e-5
        DICT_LR_FINETUNIN['7'] = 1e-5
        DICT_LR_FINETUNIN['8'] = 3e-6
        DICT_LR_FINETUNIN['9'] = 1e-6  # 0.000001
        DICT_LR_FINETUNIN['11'] = 6e-7

        train_task_one_step(model1=model1, filename_csv_train=FILENAME_CSV_TRAIN,
                            FILENAME_CSV_VALID=FILENAME_CSV_VALID, filename_csv_test=FILENAME_CSV_TEST,
                            input_shape=IMAGE_SHAPE, imgaug_train_seq=IMGAUG_TRAIN_SEQ,
                            add_top=False, change_top=True,
                            optimizer='adam', lookahead=True,
                            batch_size_train=BATCH_SIZE_TRAIN, batch_size_valid=BATCH_SIZE_TRAIN,
                            epoch_finetuning=EPOCH_FINETUNING, dict_lr_finetuning=DICT_LR_FINETUNIN,
                            class_weight=CLASS_WEIGHT, label_smoothing=LABEL_SMOOTHING,
                            weight_class_start=WEIGHT_CLASS_START, weight_class_end=WEIGHT_CLASS_START,
                            balance_ratio=BALANCE_RATIO,
                            model_save_dir=MODEL_SAVE_DIR, model_name=MODEL_NAME,
                            verbose=2, gpu_num=GPU_NUM,
                            config_file_realtime='config_file_realtime.json')
#endregion

print('OK!')


