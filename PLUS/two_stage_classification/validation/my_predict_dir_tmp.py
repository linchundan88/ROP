import os
from LIBS.DataPreprocess.my_data import get_images_labels, write_csv_dir_nolabel
from LIBS.DataValidation.my_multi_class import op_files_multiclass
import pandas as pd
from LIBS.DLP.my_predict_helper import do_predict_batch

CUDA_VISIBLE_DEVICES = "1"
GPU_NUM = 1

GEN_CSV = True
COMPUTE_DIR_FILES = True

DIR_DEST_BASE = '/media/ubuntu/data1/ROP项目/plus_combine_dataset/posterior/plus_results_b/test/0'
dir_original = '/media/ubuntu/data1/ROP项目/plus_combine_dataset/posterior/original/test/0'
dir_blood_vessel_seg = '/media/ubuntu/data1/ROP项目/plus_combine_dataset/posterior/blood_vessel_seg_result1/test/0'

filename_csv = os.path.join(DIR_DEST_BASE, 'plus_two_stage_results.csv')

if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    write_csv_dir_nolabel(filename_csv, dir_blood_vessel_seg)


dicts_models = []

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/InceptionResnetV2-008-0.958.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/InceptionV3-012-0.956.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/Xception-010-0.958.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)


df = pd.read_csv(filename_csv)
all_files, all_labels = get_images_labels(filename_csv_or_pd=df)

prob_total, y_pred_total, prob_list, pred_list = \
    do_predict_batch(dicts_models, filename_csv,
            argmax=True, cuda_visible_devices=CUDA_VISIBLE_DEVICES, gpu_num=GPU_NUM)

if COMPUTE_DIR_FILES:
    op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_blood_vessel_seg,
                        dir_dest=DIR_DEST_BASE, dir_original=dir_original, keep_subdir=True)

print('OK')