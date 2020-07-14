import os
import pandas as pd
import pickle
from LIBS.DataPreprocess.my_csv_db_helper import export_csv_from_db
from LIBS.DataPreprocess.my_data_patiend_id import write_csv_list_patient_id

#region read images and labels from database, write to csv file.
TRAIN_TYPE = 'Posterior'
DATFILE_TYPE = 'dataset5'
filename_csv = os.path.join(os.path.abspath('..'),
               'datafiles', DATFILE_TYPE, TRAIN_TYPE + '.csv')
sql = 'select SHA1,filename, posterior1 from tb_multi_labels where posterior1 is not null and uncertain is null and other_diseases is  null '
#and gradable1=1
base_dir = '/media/ubuntu/data1/ROP_dataset1/preprocess384'
export_csv_from_db(base_dir, sql, filename_csv)

#endregion

#region split dataset
filename_pkl_train = os.path.join(os.path.abspath('../../../'),
            'Data_split', 'pat_id_pkl', 'split_patid_train.pkl')
list_patient_id_train = pickle.load(open(filename_pkl_train, 'rb'))
filename_pkl_valid = os.path.join(os.path.abspath('../../../'),
            'Data_split', 'pat_id_pkl', 'split_patid_valid.pkl')
list_patient_id_valid = pickle.load(open(filename_pkl_valid, 'rb'))
filename_pkl_test = os.path.join(os.path.abspath('../../../'),
            'Data_split', 'pat_id_pkl', 'split_patid_test.pkl')
list_patient_id_test = pickle.load(open(filename_pkl_test, 'rb'))

list_patient_id_valid_test = list_patient_id_valid + list_patient_id_test
filename_csv_train = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_train.csv')
write_csv_list_patient_id(filename_csv, filename_csv_train, list_patient_id_valid_test,
                  include_or_exclude='exclude', field_columns=['images', 'labels'], del_sha1_header=True)
filename_csv_valid = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_valid.csv')
write_csv_list_patient_id(filename_csv, filename_csv_valid, list_patient_id_valid,
                          field_columns=['images', 'labels'], del_sha1_header=True)
filename_csv_test = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_test.csv')
write_csv_list_patient_id(filename_csv, filename_csv_test, list_patient_id_test,
                          field_columns=['images', 'labels'], del_sha1_header=True)

for file_csv in [filename_csv_train, filename_csv_valid, filename_csv_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    for label in [0, 1]:
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))

#endregion

print('OK')