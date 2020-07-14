import os

from LIBS.DataPreprocess.my_data import get_images_labels
from LIBS.DataValidation.my_multi_class import compute_confusion_matrix, op_files_multiclass
import pandas as pd
import pickle
from LIBS.DLP.my_predict_helper import do_predict_batch

CUDA_VISIBLE_DEVICES = "1"
GPU_NUM = 1

dir_original = '/media/ubuntu/data1/ROP_dataset/Plus/2020_02_12/original'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset/Plus/2020_02_12/blood_vessel_seg_result'

COMPUTE_CONFUSIN_MATRIX = True
COMPUTE_DIR_FILES = True

DIR_DEST_BASE = '/tmp5/Plus_steptwo_2020_4_14'
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


for predict_type_name in ['Plus_step_two_split_patid_train', 'Plus_step_two_split_patid_valid', 'Plus_step_two_split_patid_test']:

    filename_csv = os.path.join(os.path.abspath('..'),
                    'datafiles/dataset1', predict_type_name + '.csv')

    df = pd.read_csv(filename_csv)
    all_files, all_labels = get_images_labels(filename_csv_or_pd=df)

    prob_total, y_pred_total, prob_list, pred_list =\
        do_predict_batch(dicts_models, filename_csv,
                                                    argmax=True, cuda_visible_devices=CUDA_VISIBLE_DEVICES, gpu_num=GPU_NUM)

    dir_dest_confusion = os.path.join(DIR_DEST_BASE, predict_type_name, 'confusion_matrix', 'files')
    dir_dest_predict_dir = os.path.join(DIR_DEST_BASE, predict_type_name, 'dir')
    pkl_prob = os.path.join(DIR_DEST_BASE, predict_type_name + '_prob.pkl')
    pkl_confusion_matrix = os.path.join(DIR_DEST_BASE, predict_type_name + '_cf.pkl')

    os.makedirs(os.path.dirname(pkl_prob), exist_ok=True)
    with open(pkl_prob, 'wb') as file:
        pickle.dump(prob_total, file)

    # pkl_file = open(prob_pkl', 'rb')
    # prob_total = pickle.load(pkl_file)

    if COMPUTE_CONFUSIN_MATRIX:
        (cf_list, not_match_list, cf_total, not_match_total) = \
            compute_confusion_matrix(prob_list, dir_dest_confusion,
                all_files, all_labels, dir_preprocess=dir_preprocess, dir_original=dir_original)

        print(cf_total)
        os.makedirs(os.path.dirname(pkl_confusion_matrix), exist_ok=True)
        with open(pkl_confusion_matrix, 'wb') as file:
            pickle.dump(cf_total, file)

    if COMPUTE_DIR_FILES:
        op_files_multiclass(filename_csv, prob_total, dir_preprocess=dir_preprocess,
            dir_dest=dir_dest_predict_dir, dir_original=dir_original, keep_subdir=True)


print('OK')