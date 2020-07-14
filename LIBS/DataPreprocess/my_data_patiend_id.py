import math
import os
import random

import pandas as pd
import sklearn
import csv

def split_dataset_by_pat_id(filename_csv_or_df,
                valid_ratio=0.1, test_ratio=None, shuffle=True, random_state=None):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    list_patient_id = []

    for _, row in df.iterrows():
        image_file = row['images']
        _, filename = os.path.split(image_file)

        pat_id = filename.split('.')[0]
        if pat_id not in list_patient_id:
            list_patient_id.append(pat_id)

    list_patient_id = sklearn.utils.shuffle(list_patient_id, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        train_files = []
        train_labels = []
        valid_files = []
        valid_labels = []

        for _, row in df.iterrows():
            image_file = row['images']
            image_labels = row['labels']
            _, filename = os.path.split(image_file)

            pat_id = filename.split('.')[0]

            if pat_id in list_patient_id_train:
                train_files.append(image_file)
                train_labels.append(image_labels)

            if pat_id in list_patient_id_valid:
                valid_files.append(image_file)
                valid_labels.append(image_labels)

        return train_files, train_labels, valid_files, valid_labels

    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        train_files = []
        train_labels = []
        valid_files = []
        valid_labels = []
        test_files = []
        test_labels = []

        for _, row in df.iterrows():
            image_file = row['images']
            image_labels = row['labels']
            _, filename = os.path.split(image_file)

            pat_id = filename.split('.')[0]

            if pat_id in list_patient_id_train:
                train_files.append(image_file)
                train_labels.append(image_labels)

            if pat_id in list_patient_id_valid:
                valid_files.append(image_file)
                valid_labels.append(image_labels)

            if pat_id in list_patient_id_test:
                test_files.append(image_file)
                test_labels.append(image_labels)

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels


def get_list_patient_id(dir, del_sha1_header=False):
    list_patient_id = []

    for dir_path, subpaths, files in os.walk(dir, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)
            if not '/original/' in img_file_source:
                continue

            _, filename = os.path.split(img_file_source)
            file_basename, file_extension = os.path.splitext(filename)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF', '.PPF']:
                # print('file ext name:', f)
                continue

            if del_sha1_header:
                # 0a1d0ef72e0bc69c01f87502faba2524fbd7d7b6#a9c63c7b-32f7-4a41-bc94-03469fd8c7e0.14.jpg
                file_basename = file_basename.split('#')[1]

            # '0f104b1b-d1cd-4acc-8a1b-4e88c61d18b0.16.jpg'
            patient_id = file_basename.split('.')[0]
            if patient_id not in list_patient_id:
                list_patient_id.append(patient_id)

    return list_patient_id


def split_list_patient_id(list_patient_id, valid_ratio=0.1, test_ratio=0.1,
                          random_seed=18888):
    random.seed(random_seed)
    random.shuffle(list_patient_id)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        return list_patient_id_train, list_patient_id_valid
    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        return list_patient_id_train, list_patient_id_valid, list_patient_id_test


def split_dataset_by_predefined_pat_id(filename_csv_or_df,
                list_patient_id_train, list_patient_id_valid, list_patient_id_test):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    train_files = []
    train_labels = []
    valid_files = []
    valid_labels = []
    test_files = []
    test_labels = []

    for _, row in df.iterrows():
        image_file = row['images']
        image_labels = row['labels']
        _, filename = os.path.split(image_file)

        pat_id = filename.split('.')[0]

        if pat_id in list_patient_id_train:
            train_files.append(image_file)
            train_labels.append(image_labels)

        if pat_id in list_patient_id_valid:
            valid_files.append(image_file)
            valid_labels.append(image_labels)

        if pat_id in list_patient_id_test:
            test_files.append(image_file)
            test_labels.append(image_labels)

    return train_files, train_labels, valid_files, valid_labels, test_files, test_labels


def write_csv_list_patient_id(filename_csv, filename_csv_new,
        list_patient_id,  include_or_exclude='include', field_columns=['images', 'labels'],
        del_sha1_header=True):

    if os.path.exists(filename_csv_new):
        os.remove(filename_csv_new)
    os.makedirs(os.path.dirname(filename_csv_new), exist_ok=True)

    with open(filename_csv_new, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow([field_columns[0], field_columns[1]])

        df = pd.read_csv(filename_csv)
        for _, row in df.iterrows():
            image_file = row[field_columns[0]]
            image_label = row[field_columns[1]]

            _, filename = os.path.split(image_file)
            filename_base, filename_ext = os.path.splitext(filename)

            if del_sha1_header:
                # 0a1d0ef72e0bc69c01f87502faba2524fbd7d7b6#a9c63c7b-32f7-4a41-bc94-03469fd8c7e0.14.jpg
                # prob50.1#0a1d0ef72e0bc69c01f87502faba2524fbd7d7b6#a9c63c7b-32f7-4a41-bc94-03469fd8c7e0.14.jpg
                if len(filename_base.split('#')) == 1:
                    filename_base = filename_base.split('#')[0]
                if len(filename_base.split('#')) == 2:
                    filename_base = filename_base.split('#')[1]
                if len(filename_base.split('#')) == 3:
                    filename_base = filename_base.split('#')[2]

            patient_id = filename_base.split('.')[0]
            assert include_or_exclude in ['include', 'exclude'], 'include_or_exclude error'
            if include_or_exclude == 'include':
                if patient_id in list_patient_id:
                    csv_writer.writerow([image_file, image_label])
            elif include_or_exclude == 'exclude':
                if patient_id not in list_patient_id:
                    csv_writer.writerow([image_file, image_label])

    print('write csv ok!')


def split_dataset_by_pat_id_cross_validation(filename_csv_or_df, shuffle=True,
             num_cross_validation=5, random_state=None):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    list_patient_id = []
    for _, row in df.iterrows():
        image_file = row['images']
        _, filename = os.path.split(image_file)
        pat_id = filename.split('.')[0]
        if pat_id not in list_patient_id:
            list_patient_id.append(pat_id)

    train_files = [[] for _ in range(num_cross_validation)]
    train_labels = [[] for _ in range(num_cross_validation)]
    valid_files = [[] for _ in range(num_cross_validation)]
    valid_labels = [[] for _ in range(num_cross_validation)]

    for i in range(num_cross_validation):
        patient_id_batch = math.ceil(len(list_patient_id) / num_cross_validation)

        list_patient_id_valid = list_patient_id[i*patient_id_batch: (i+1)*patient_id_batch]

        list_patient_id_train = []
        for pat_id in list_patient_id:
            if pat_id not in list_patient_id_valid:
                list_patient_id_train.append(pat_id)

        for _, row in df.iterrows():
            image_file = row['images']
            image_labels = row['labels']
            _, filename = os.path.split(image_file)

            pat_id = filename.split('.')[0]

            if pat_id in list_patient_id_train:
                train_files[i].append(image_file)
                train_labels[i].append(image_labels)

            if pat_id in list_patient_id_valid:
                valid_files[i].append(image_file)
                valid_labels[i].append(image_labels)

    return train_files, train_labels, valid_files, valid_labels