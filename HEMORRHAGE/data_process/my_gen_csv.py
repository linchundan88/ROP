import os
import pandas as pd
import pickle
from LIBS.DataPreprocess.my_data import write_csv_based_on_dir
from LIBS.DataPreprocess.my_data_patiend_id import write_csv_list_patient_id

#region read files, extract labels based on subdirectorie names, write to csv file.
TRAIN_TYPE = 'Hemorrhage'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset/Hemorrhage/preprocess384/'
DATFILE_TYPE = 'dataset7'
filename_csv = os.path.join(os.path.abspath('..'),
               'datafiles', DATFILE_TYPE, TRAIN_TYPE + '.csv')
dict_mapping = {'0': 0, '对照组': 0, '1': 1, '出血': 1}
write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping, match_type='partial')

#endregion

#region split dataset

filename_pkl_train = os.path.join(os.path.abspath('../../'),
         'Data_split', 'pat_id_pkl', 'split_patid_train.pkl')
list_patient_id_train = pickle.load(open(filename_pkl_train, 'rb'))
filename_pkl_valid = os.path.join(os.path.abspath('../../'),
        'Data_split', 'pat_id_pkl', 'split_patid_valid.pkl')
list_patient_id_valid = pickle.load(open(filename_pkl_valid, 'rb'))
filename_pkl_test = os.path.join(os.path.abspath('../../'),
        'Data_split', 'pat_id_pkl', 'split_patid_test.pkl')
list_patient_id_test = pickle.load(open(filename_pkl_test, 'rb'))

filename_csv_train = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_train.csv')
write_csv_list_patient_id(filename_csv, filename_csv_train, list_patient_id_train,
                          field_columns=['images', 'labels'])
filename_csv_valid = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_valid.csv')
write_csv_list_patient_id(filename_csv, filename_csv_valid, list_patient_id_valid,
                          field_columns=['images', 'labels'])
filename_csv_test = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_test.csv')
write_csv_list_patient_id(filename_csv, filename_csv_test, list_patient_id_test,
                          field_columns=['images', 'labels'])


for file_csv in [filename_csv, filename_csv_train, filename_csv_valid, filename_csv_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    num_class = df['labels'].nunique(dropna=True)
    for label in range(num_class):
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))

#endregion


print('OK')