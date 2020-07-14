
import  os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
GPU_NUM = 3
import numpy as np
import collections
from LIBS.DLP.my_train_multiclass_helper import train_task_two_steps

#region train parameters
TRAIN_TYPE = 'Stage'
DATFILE_TYPE = 'dataset4'
FILENAME_CSV_TRAIN = os.path.join(os.path.abspath('..'),
              'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_train.csv')
FILENAME_CSV_VALID = os.path.join(os.path.abspath('..'),
              'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_valid.csv')
FILENAME_CSV_TEST = os.path.join(os.path.abspath('..'),
              'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_test.csv')
MODEL_SAVE_BASEDIR = os.path.join('/tmp3/models_ROP_2020_01_03/', TRAIN_TYPE)

from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.96, aug)
# sometimes1 = lambda aug: iaa.Sometimes(0.96, aug)
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
14928
0 13486
1 1442
2099
0 1887
1 212
'''

LABEL_SMOOTHING = 0.1

CLASS_WEIGHT = {0: 1., 1: 5}

WEIGHT_CLASS_START = np.array([1, 3])
WEIGHT_CLASS_END = np.array([1, 3])
BALANCE_RATIO = 0.93
#endregion

#region training
OPTIMIZER = 'adam' # 'adam' 'SGD', 'adabound'
LOOKAHEAD = True

TRAIN_TIMES = 5
FINE_TUNING_TIMES = 6
TRAIN_MODE_TYPES = ['InceptionV3']

for i in range(TRAIN_TIMES):
    for j in range(FINE_TUNING_TIMES):

        MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_BASEDIR, 'train' + str(i), 'transfer' + str(j))

        #region InceptionV3
        MODEL_NAME = 'InceptionV3'
        IMAGE_SHAPE = (299, 299, 3)
        if MODEL_NAME in TRAIN_MODE_TYPES:
            model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'

            DICT_LR_TRAINTOP = collections.OrderedDict()
            DICT_LR_TRAINTOP['0'] = 1e-3
            DICT_LR_TRAINTOP['2'] = 3e-4
            DICT_LR_TRAINTOP['3'] = 1e-4

            DICT_LR_FINETUNING = collections.OrderedDict()
            DICT_LR_FINETUNING['0'] = 1e-4
            DICT_LR_FINETUNING['2'] = 1e-5
            DICT_LR_FINETUNING['4'] = 1e-6  # 0.000001
            DICT_LR_FINETUNING['6'] = 6e-7

            #total 313 layer  freeze_layes_num = None #GAP layer
            # 100 mixed3,
            # 132 mixed4
            # 164 mixed5
            # 196 mixed6
            # 228 mixed7
            # 248 mixed8
            # 279 mixed9
            # 310 mixed10

            LIST_FINETUNING_LAYERS = [210, 220, 230, 250, 270, 290]
            # LIST_FINETUNING_LAYERS = [164, 196, 228, 248, 279, 310]
            FREEZE_LAYERS_NUM = LIST_FINETUNING_LAYERS[j]

            if j == 0:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 1:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 2:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 3:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8
            if j == 4:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8
            if j == 5:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8

            train_task_two_steps(model1=model_file, filename_csv_train=FILENAME_CSV_TRAIN,
                                 FILENAME_CSV_VALID=FILENAME_CSV_VALID, filename_csv_test=FILENAME_CSV_TEST,
                                 input_shape=IMAGE_SHAPE, imgaug_train_seq=IMGAUG_TRAIN_SEQ,
                                 add_top=False, change_top=True, freeze_layes_num=FREEZE_LAYERS_NUM,
                                 optimizer=OPTIMIZER, lookahead=LOOKAHEAD,
                                 epoch_traintop=EPOCH_TRAINTOP, epoch_finetuning=EPOCH_FINETUNING,
                                 dict_lr_traintop=DICT_LR_TRAINTOP, dict_lr_finetuning=DICT_LR_FINETUNING,
                                 class_weight=CLASS_WEIGHT, label_smoothing=LABEL_SMOOTHING,
                                 weight_class_start=WEIGHT_CLASS_START, weight_class_end=WEIGHT_CLASS_END,
                                 balance_ratio=BALANCE_RATIO,
                                 model_save_dir=MODEL_SAVE_DIR, model_name=MODEL_NAME,
                                 verbose=2, gpu_num=GPU_NUM,
                                 config_file_realtime='config_file_realtime.json')
        #endregion

        #region Xception
        MODEL_NAME = 'Xception'
        IMAGE_SHAPE = (299, 299, 3)
        if MODEL_NAME in TRAIN_MODE_TYPES:
            # model_file = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/Multi_label_Xception-015-train0.9671_val0.945.hdf5'
            # model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/Transfer_learning/Xception-008-0.957.hdf5'
            model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/2019_4_19/split_pat_id/Inception_V3-006-0.955.hdf5'

            #total 134 layer  freeze_layes_num = None #GAP layer
            LIST_FINETUNING_LAYERS = [100, 105, 110, 115, 120, 125]
            FREEZE_LAYERS_NUM = LIST_FINETUNING_LAYERS[j]

            if j == 0:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 1:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 2:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 3:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8
            if j == 4:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8
            if j == 5:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8

            DICT_LR_TRAINTOP = collections.OrderedDict()
            DICT_LR_TRAINTOP['0'] = 1e-3
            DICT_LR_TRAINTOP['2'] = 3e-4
            DICT_LR_TRAINTOP['3'] = 1e-4

            DICT_LR_FINETUNING = collections.OrderedDict()

            DICT_LR_FINETUNING['0'] = 1e-4
            DICT_LR_FINETUNING['2'] = 1e-5
            DICT_LR_FINETUNING['4'] = 1e-6  # 0.000001
            DICT_LR_FINETUNING['6'] = 6e-7

            train_task_two_steps(model1=model_file, filename_csv_train=FILENAME_CSV_TRAIN,
                                 FILENAME_CSV_VALID=FILENAME_CSV_VALID, filename_csv_test=FILENAME_CSV_TEST,
                                 input_shape=IMAGE_SHAPE, imgaug_train_seq=IMGAUG_TRAIN_SEQ,
                                 add_top=False, change_top=True, freeze_layes_num=FREEZE_LAYERS_NUM,
                                 optimizer=OPTIMIZER, lookahead=LOOKAHEAD,
                                 epoch_traintop=EPOCH_TRAINTOP, epoch_finetuning=EPOCH_FINETUNING,
                                 dict_lr_traintop=DICT_LR_TRAINTOP, dict_lr_finetuning=DICT_LR_FINETUNING,
                                 class_weight=CLASS_WEIGHT, label_smoothing=LABEL_SMOOTHING,
                                 weight_class_start=WEIGHT_CLASS_START, weight_class_end=WEIGHT_CLASS_END,
                                 balance_ratio=BALANCE_RATIO,
                                 model_save_dir=MODEL_SAVE_DIR, model_name=MODEL_NAME,
                                 verbose=2, gpu_num=GPU_NUM,
                                 config_file_realtime='config_file_realtime.json')
        # endregion

        #region InceptionResnetV2
        MODEL_NAME = 'InceptionResNetV2'
        IMAGE_SHAPE = (299, 299, 3)
        if MODEL_NAME in TRAIN_MODE_TYPES:
            # model_file = '/home/ubuntu/dlp/deploy_models_new/bigclasses_multilabels/class_weights5_0.2_0.7/Multi_label_InceptionResNetV2-006-train0.9674_val0.951.hdf5'
            model_file = '/home/ubuntu/dlp/deploy_models_2019/bigclass_multiclass/Transfer_learning/InceptionResnetV2-006-0.962.hdf5'

            DICT_LR_TRAINTOP = collections.OrderedDict()
            DICT_LR_TRAINTOP['0'] = 1e-3
            DICT_LR_TRAINTOP['2'] = 3e-4
            DICT_LR_TRAINTOP['3'] = 1e-4

            DICT_LR_FINETUNING = collections.OrderedDict()
            DICT_LR_FINETUNING['0'] = 1e-4
            DICT_LR_FINETUNING['1'] = 1e-5
            DICT_LR_FINETUNING['3'] = 1e-6  # 0.000001
            DICT_LR_FINETUNING['5'] = 6e-7

            # total layers 782
            LIST_FINETUNING_LAYERS = [560, 600, 650, 700, 725, 750]
            FREEZE_LAYERS_NUM = LIST_FINETUNING_LAYERS[j]

            if j == 0:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 1:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 2:
                EPOCH_TRAINTOP = 6
                EPOCH_FINETUNING = 8
            if j == 3:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8
            if j == 4:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8
            if j == 5:
                EPOCH_TRAINTOP = 5
                EPOCH_FINETUNING = 8

            train_task_two_steps(model1=model_file, filename_csv_train=FILENAME_CSV_TRAIN,
                                 FILENAME_CSV_VALID=FILENAME_CSV_VALID, filename_csv_test=FILENAME_CSV_TEST,
                                 input_shape=IMAGE_SHAPE, imgaug_train_seq=IMGAUG_TRAIN_SEQ,
                                 add_top=False, change_top=True, freeze_layes_num=FREEZE_LAYERS_NUM,
                                 optimizer=OPTIMIZER, lookahead=LOOKAHEAD,
                                 epoch_traintop=EPOCH_TRAINTOP, epoch_finetuning=EPOCH_FINETUNING,
                                 dict_lr_traintop=DICT_LR_TRAINTOP, dict_lr_finetuning=DICT_LR_FINETUNING,
                                 class_weight=CLASS_WEIGHT, label_smoothing=LABEL_SMOOTHING,
                                 weight_class_start=WEIGHT_CLASS_START, weight_class_end=WEIGHT_CLASS_END,
                                 balance_ratio=BALANCE_RATIO,
                                 model_save_dir=MODEL_SAVE_DIR, model_name=MODEL_NAME,
                                 verbose=2, gpu_num=GPU_NUM,
                                 config_file_realtime='config_file_realtime.json')

        #endregion


#endregion

print('OK!')

