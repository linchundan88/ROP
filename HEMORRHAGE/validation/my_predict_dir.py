import os
from LIBS.DataPreprocess.my_data import get_images_labels,write_csv_dir_nolabel
from LIBS.DataValidation.my_multi_class import op_files_multiclass
import pandas as pd
from LIBS.DLP.my_predict_helper import do_predict_batch

CUDA_VISIBLE_DEVICES = "2"
GPU_NUM = 1

DO_PREPROCESS = True
GEN_CSV = True
COMPUTE_DIR_FILES = True

dir_original = '/media/ubuntu/data1/ROP项目/人机比赛用图_20200317/original'
dir_preprocess = '/media/ubuntu/data1/ROP项目/人机比赛用图_20200317/preprocess384/'
dir_dest = '/media/ubuntu/data1/ROP项目/人机比赛用图_20200317/results_2020_5_20/Hemorrhage'
pkl_prob = os.path.join(dir_dest, 'probs.pkl')

from LIBS.ImgPreprocess import my_preprocess_dir
if DO_PREPROCESS:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
        image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

filename_csv = os.path.join(dir_dest, 'hemorrhage_predict_dir.csv')
if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    write_csv_dir_nolabel(filename_csv, dir_preprocess)


dicts_models = []

# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_3_7/InceptionResnetV2-007-0.993.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)
#
# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_3_7/InceptionV3-008-0.991.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)
#
# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_3_7/Xception-005-0.989.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_5_19/InceptionResnetV2-007-0.990.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_5_19/InceptionV3-005-0.988.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_5_19/Xception-007-0.988.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

df = pd.read_csv(filename_csv)
all_files, all_labels = get_images_labels(filename_csv_or_pd=df)

prob_total, y_pred_total, prob_list, pred_list = \
    do_predict_batch(dicts_models, filename_csv,
            argmax=True, cuda_visible_devices=CUDA_VISIBLE_DEVICES, gpu_num=GPU_NUM)

import pickle
os.makedirs(os.path.dirname(pkl_prob), exist_ok=True)
with open(pkl_prob, 'wb') as file:
    pickle.dump(prob_total, file)

if COMPUTE_DIR_FILES:
    op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
                        dir_dest=dir_dest, dir_original=dir_original, keep_subdir=True)

print('OK')