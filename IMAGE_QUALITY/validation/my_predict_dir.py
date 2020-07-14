import os
from LIBS.DataPreprocess.my_data import get_images_labels,write_csv_dir_nolabel
from LIBS.DataValidation.my_multi_class import op_files_multiclass
import pandas as pd
from LIBS.DLP.my_predict_helper import do_predict_batch

CUDA_VISIBLE_DEVICES = "1"
GPU_NUM = 1

DO_PREPROCESS = True
GEN_CSV = True
COMPUTE_DIR_FILES = True

dir_original = '/media/ubuntu/data1/ROP_dataset1/original'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset1/preprocess384'
dir_dest = '/media/ubuntu/data2/tmp5/2020_3_13_results/image_quality'
pkl_prob = os.path.join(dir_dest, 'probs.pkl')

from LIBS.ImgPreprocess import my_preprocess_dir
if DO_PREPROCESS:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
                                        image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

filename_csv = os.path.join(dir_dest, 'image_quality_predict_dir.csv')
if GEN_CSV:
    if not os.path.exists(os.path.dirname(filename_csv)):
        os.makedirs(os.path.dirname(filename_csv))

    write_csv_dir_nolabel(filename_csv, dir_preprocess)


dicts_models = []

model_dir = '/home/ubuntu/dlp/deploy_models/ROP/image_quality/2020_2_24'
dicts_models = []

dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionV3-009-0.953.hdf5'),
                'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)
dict_model1 = {'model_file': os.path.join(model_dir, 'Xception-010-0.957.hdf5'),
                'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)
dict_model1 = {'model_file': os.path.join(model_dir, 'NASNetMobile-008-0.959.hdf5'),
                'input_shape': (224, 224, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

df = pd.read_csv(filename_csv)
all_files, all_labels = get_images_labels(filename_csv_or_pd=df)

prob_total, y_pred_total, prob_list, pred_list = \
    do_predict_batch(dicts_models, filename_csv,
            argmax=True, cuda_visible_devices=CUDA_VISIBLE_DEVICES, gpu_num=GPU_NUM)

import pickle
if not os.path.exists(os.path.dirname(pkl_prob)):
    os.makedirs(os.path.dirname(pkl_prob))
with open(pkl_prob, 'wb') as file:
    pickle.dump(prob_total, file)

if COMPUTE_DIR_FILES:
    op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
                        dir_dest=dir_dest, dir_original=dir_original, keep_subdir=True)

print('OK')