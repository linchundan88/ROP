import os
import sys
from LIBS.DataPreprocess import my_data
import pandas as pd
import numpy as np
import pickle


# filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
#                 'datafiles', 'Grade_split_patid_train.csv'))
filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
                'datafiles', 'Grade_split_patid_valid.csv'))


pkl_file = open('/tmp3/ROP_NEW/Grade/pat_id_split_2019_10_8/Grade_split_patid_valid_prob.pkl', 'rb')
probs = pickle.load(pkl_file)


df = pd.read_csv(filename_csv)
all_files, all_labels = my_data.get_images_labels(filename_csv_or_pd=df)

dict_patient_gt = {}
dict_patient_predict = {}

for i, row in df.iterrows():
    image_file = row['images']
    labels = int(row['labels'])

    _, filename = os.path.split(image_file)
    pat_id = filename.split('.')[0]

    if pat_id not in dict_patient_gt:
        dict_patient_gt[pat_id] = 0
    elif dict_patient_gt[pat_id] == 0 and labels == 1:
        dict_patient_gt[pat_id] = 1

    if pat_id not in dict_patient_predict:
        dict_patient_predict[pat_id] = 0
    elif dict_patient_predict[pat_id] == 0 and np.argmax(probs[i]) == 1:
        dict_patient_predict[pat_id] = 1


list_label_gt = []
list_label_predict = []

for (k, v) in dict_patient_gt.items():
    list_label_gt.append(v)
    list_label_predict.append(dict_patient_predict[k])

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

labels = [x for x in range(0, 2)]
confusion_matrix = sk_confusion_matrix(list_label_gt, list_label_predict, labels=labels)

print('OK')


'''
train
0 = [1256, 43]
1 = [9, 249]

validation
0 = [150, 8]
1 = [1, 34]

'''
