import os
import pickle
from LIBS.DataPreprocess import my_data
import pandas as pd
from LIBS.DataPreprocess.my_data_patiend_id import write_csv_list_patient_id

#region read files, extract labels based on subdirectorie names, write to csv file.
TRAIN_TYPE = 'Plus_step_two'
dir_preprocess = '/media/ubuntu/data2/plus_2020_5_22/blood_vessel_seg_posterior'
DATFILE_TYPE = 'dataset7'
filename_csv = os.path.join(os.path.abspath('..'),
               'datafiles', DATFILE_TYPE, TRAIN_TYPE + '.csv')
dict_mapping = {'0': 0, '正常': 0, '对照': 0, '1': 1, 'PLUS对照': 0, 'PLUS': 1}
my_data.write_csv_based_on_dir(filename_csv, dir_preprocess, dict_mapping, match_type='partial')

#endregion

filename_pkl_train = os.path.join(os.path.abspath('../../../../'),
         'Data_split', 'pat_id_pkl', 'split_patid_train.pkl')
list_patient_id_train = pickle.load(open(filename_pkl_train, 'rb'))
filename_pkl_valid = os.path.join(os.path.abspath('../../../../'),
        'Data_split', 'pat_id_pkl', 'split_patid_valid.pkl')
list_patient_id_valid = pickle.load(open(filename_pkl_valid, 'rb'))
filename_pkl_test = os.path.join(os.path.abspath('../../../../'),
        'Data_split', 'pat_id_pkl', 'split_patid_test.pkl')
list_patient_id_test = pickle.load(open(filename_pkl_test, 'rb'))


filename_csv_train = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_train.csv')
write_csv_list_patient_id(filename_csv, filename_csv_train, list_patient_id_train,
                          field_columns=['images', 'labels'])
FILENAME_CSV_VALID = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_valid.csv')
write_csv_list_patient_id(filename_csv, FILENAME_CSV_VALID, list_patient_id_valid,
                          field_columns=['images', 'labels'])

filename_csv_test = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_test.csv')
write_csv_list_patient_id(filename_csv, filename_csv_test, list_patient_id_test,
                          field_columns=['images', 'labels'])

for file_csv in [filename_csv_train, FILENAME_CSV_VALID, filename_csv_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    for label in [0, 1]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))



#endregion


print('OK')