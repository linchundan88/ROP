import os
from LIBS.DataPreprocess import my_data
import pandas as pd
import pickle
from LIBS.DataPreprocess.my_data import write_csv_based_on_dir
from LIBS.DataPreprocess.my_data_patiend_id import write_csv_list_patient_id

#region read files, extract labels based on subdirectorie names, write to csv file.
TRAIN_TYPE = 'Stage'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset/Stage/preprocess384/'
DATFILE_TYPE = 'dataset11'
filename_csv = os.path.join(os.path.abspath('..'),
               'datafiles', DATFILE_TYPE, TRAIN_TYPE + '.csv')

dict_mapping = {'0': 0, '正常_无反光': 0, '正常_反光': 0, '分期病变对照': 0, '分期病变': 1, '1': 1}
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
filename_cvs_valid = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_valid.csv')
write_csv_list_patient_id(filename_csv, filename_cvs_valid, list_patient_id_valid,
                          field_columns=['images', 'labels'])
filename_csv_test = os.path.join(os.path.abspath('..'),
                    'datafiles', DATFILE_TYPE, TRAIN_TYPE + '_split_patid_test.csv')
write_csv_list_patient_id(filename_csv, filename_csv_test, list_patient_id_test,
                          field_columns=['images', 'labels'])

for file_csv in [filename_csv_train, filename_cvs_valid, filename_csv_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    num_class = df['labels'].nunique(dropna=True)
    for label in range(num_class):
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))


exit(0)
#endregion


#region split dataset,  5 fold Cross validation
'''
list_train_files, list_train_labels, list_valid_files, list_valid_labels =\
    LIBS.DataPreprocess.my_data_patiend_id.split_dataset_by_pat_id_cross_validation(filename_csv,
                                                                                    num_cross_validation=7, random_state=222)

for i in range(len(list_train_files)):
    my_data.write_images_labels_csv(list_train_files[i], list_train_labels[i],
                                    filename_csv=os.path.join(os.path.abspath('..'),
           'datafiles', "{}_{}_{} {}".format(TRAIN_TYPE, 'train', i, '.csv')))
    my_data.write_images_labels_csv(list_valid_files[i], list_valid_labels[i],
                                    filename_csv=os.path.join(os.path.abspath('..'),
            'datafiles', "{}_{}_{} {}".format(TRAIN_TYPE, 'valid', i, '.csv')))

'''
#endregion


print('OK')