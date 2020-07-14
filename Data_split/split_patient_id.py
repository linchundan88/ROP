import os
import pickle

from LIBS.DataPreprocess.my_data_patiend_id import get_list_patient_id, \
    split_list_patient_id

dir_path = '/media/ubuntu/data1/ROP_dataset1/original'
list_patient_id = get_list_patient_id(dir_path,  del_sha1_header=True)
list_patient_id_train, list_patient_id_valid, list_patient_id_test\
    = split_list_patient_id(list_patient_id, valid_ratio=0.1, test_ratio=0.15, random_seed=48888)


os.makedirs(os.path.join(os.path.abspath('.'), 'pat_id_pkl'), exist_ok=True)
filename_pkl_train = os.path.join(os.path.abspath('.'),
                    'pat_id_pkl', 'split_patid_train.pkl')
with open(filename_pkl_train, 'wb') as file:
    pickle.dump(list_patient_id_train, file)

filename_pkl_valid = os.path.join(os.path.abspath('.'),
                    'pat_id_pkl', 'split_patid_valid.pkl')
with open(filename_pkl_valid, 'wb') as file:
    pickle.dump(list_patient_id_valid, file)

filename_pkl_test = os.path.join(os.path.abspath('.'),
                    'pat_id_pkl', 'split_patid_test.pkl')
with open(filename_pkl_test, 'wb') as file:
    pickle.dump(list_patient_id_test, file)

print('OK')