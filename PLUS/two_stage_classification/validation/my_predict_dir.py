import os
from LIBS.DataPreprocess.my_data import get_images_labels, write_csv_dir_nolabel
from LIBS.DataValidation.my_multi_class import op_files_multiclass
import pandas as pd
from LIBS.DLP.my_predict_helper import do_predict_batch

CUDA_VISIBLE_DEVICES = "0"
GPU_NUM = 1

GEN_CSV = True
COMPUTE_DIR_FILES = True

dir_original = '/media/ubuntu/data1/ROP项目/人机比赛用图_20200317/original/三标签'
dir_blood_vessel_seg = '/media/ubuntu/data1/ROP项目/人机比赛用图_20200317/results/Plus/blood_vessel_seg'
dir_dest = '/media/ubuntu/data1/ROP项目/人机比赛用图_20200317/results_2020_5_20/Plus/result_2020_5_20'

# dir_original = '/tmp5/ROP_human_AI/mydataset/正常/original'
# dir_blood_vessel_seg = '/tmp5/ROP_human_AI/mydataset/正常/blood_vessel_seg_result'
# dir_dest = '/tmp5/ROP_human_AI/mydataset/正常/Plus_two_steps'
pkl_prob = os.path.join(dir_dest, 'probs.pkl')

filename_csv = os.path.join(dir_dest, 'plus_two_stage_results.csv')

if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    write_csv_dir_nolabel(filename_csv, dir_blood_vessel_seg)


# dicts_models = []
# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_4_28/InceptionResnetV2-008-0.973.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)
# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_4_28/InceptionV3-006-0.969.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)
# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_4_28/Xception-005-0.967.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)

dicts_models = []
model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_5_19/InceptionResnetV2-005-0.972.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)
model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_5_19/InceptionV3-008-0.979.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)
model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_5_19/Xception-008-0.979.hdf5'
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
    op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_blood_vessel_seg,
                        dir_dest=dir_dest, dir_original=dir_original, keep_subdir=True)

print('OK')

