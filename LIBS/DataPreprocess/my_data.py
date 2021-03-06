'''

write_csv_based_on_dir
  generate labels based on dir, and write images and labels to csv file


split_dataset   (split dataset into training and validation datasets)
split_dataset_cross_validation

split_dataset_by_pat_id
  (split dataset into training and validation datasets, and suppose pat_id is the prefix of image filename)
split_dataset_by_pat_id_cross_validation


split_images_masks(image segmentation)


write_csv_files() write list images and labels to csv file
  image_files and labels after split(based on pat_id or not), write to csv files.



get_images_labels 分类 获取文件名 和 标注类别， 用于计算confusion_matrix等验证

'''


import os
import csv
import pandas as pd
import sklearn
import math


#region image two_stage_classification training

def write_csv_based_on_dir(filename_csv, base_dir, dict_mapping, match_type='header',
       list_file_ext=['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']):

    assert match_type in ['header', 'partial', 'end'], 'match type is error'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, subpaths, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                (filedir, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)

                if extension.upper() not in list_file_ext:
                    print('file ext name:', f)
                    continue

                if not filedir.endswith('/'):
                    filedir += '/'

                for (k, v) in dict_mapping.items():
                    if match_type == 'header':
                        dir1 = os.path.join(base_dir, k)
                        if not dir1.endswith('/'):
                            dir1 += '/'

                        if dir1 in filedir:
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'partial':
                        if '/' + k + '/' in filedir:
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'end':
                        if filedir.endswith('/' + k + '/'):
                            csv_writer.writerow([img_file_source, v])
                            break


#读取csv文件，分割成训练集和验证集，返回训练集图像文件和标注，以及验证集图像文件和标注
#dir_orig, dir_dest 当csv 目录不一致的时候用
def split_dataset(filename_csv_or_df, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'labels']):

    if isinstance(filename_csv_or_df, str):
        if filename_csv_or_df.endswith('.csv'):
            df = pd.read_csv(filename_csv_or_df)
        elif filename_csv_or_df.endswith('.xls') or filename_csv_or_df.endswith('.xlsx'):
            df = pd.read_excel(filename_csv_or_df)
    else:
        df = filename_csv_or_df

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df)*(1-valid_ratio))
        data_train = df[:split_num]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        data_valid = df[split_num:]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        data_test = df[split_num_valid:]
        test_files = data_test[field_columns[0]].tolist()
        test_labels = data_test[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels


def split_dataset_cross_validation(filename_csv_or_df, shuffle=True,
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


    train_files = [[] for _ in range(num_cross_validation)]
    train_labels = [[] for _ in range(num_cross_validation)]
    valid_files = [[] for _ in range(num_cross_validation)]
    valid_labels = [[] for _ in range(num_cross_validation)]

    batch_cross_validation = math.ceil(len(df) / num_cross_validation)

    for i in range(num_cross_validation):

        for j in range(len(df)):
            image_file = df.iat[i, 0]
            image_labels = df.iat[i, 1]

            if j >= i * batch_cross_validation and j<(i + 1) * batch_cross_validation:
                train_files[i].append(image_file)
                train_labels[i].append(image_labels)
            else:
                valid_files[i].append(image_file)
                valid_labels[i].append(image_labels)


        train_files[i] = train_files.tolist()
        train_labels[i] = train_labels.tolist()

        valid_files[i] = valid_files.tolist()
        valid_files[i] = valid_labels.tolist()

    return train_files, train_labels, valid_files, valid_labels


def write_images_labels_csv(files, labels, filename_csv, field_columns=['images', 'labels']):
    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow([field_columns[0], field_columns[1]])

        for i, file in enumerate(files):
            csv_writer.writerow([file, labels[i]])

    print('write csv ok!')


#endregion

#region image  two_stage_classification, validation

# get_list_from_dir and write to csv,  label all 0, used for validation
def write_csv_dir_nolabel(filename_csv, base_dir, replace_dir=False):
    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, subpaths, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                (filepath, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)

                if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                    continue

                if replace_dir:  # remove base dir
                    img_file_source = img_file_source.replace(base_dir, '')

                csv_writer.writerow([img_file_source, 0])


#批量计算(例如confusion_matrix)时候用，全部数据，不拆分  没必要shuffle，保留也无所谓
def get_images_labels(filename_csv_or_pd, shuffle=False):
    if isinstance(filename_csv_or_pd, str):
        df = pd.read_csv(filename_csv_or_pd)
    else:
        df = filename_csv_or_pd

    if shuffle:
        df = sklearn.utils.shuffle(df)

    data_all_image_file = df['images']
    data_all_labels = df['labels']

    all_files = data_all_image_file.tolist()
    all_labels = data_all_labels.tolist()

    return all_files, all_labels

#endregion


''' 

# custom defined acc
def my_acc(model, NUM_CLASSES , valid_files, valid_labels, neglect_special=True):
    # valid_files, valid_labels
    prob = np.empty((0, NUM_CLASSES))
    for x in my_Generator_test(valid_files, batch_size=64):
        probabilities = model.predict_on_batch(x)
        if prob.size == 0:
            prob = probabilities
        else:
            prob = np.vstack((prob, probabilities))

    y_pred = prob.argmax(axis=-1)

    num_same = 0
    for i in range(len(valid_labels)):
        if valid_labels[i] == y_pred[i]:
            num_same = num_same + 1

        if neglect_special:
            # normal:1 Tigroid fundus:2
            if valid_labels[i] == 1 and y_pred[i] == 2:
                num_same = num_same + 1
            if valid_labels[i] == 2 and y_pred[i] == 1:
                num_same = num_same + 1

            # 屈光介质混浊:34  屈光介质混浊-增殖性DR可能性大:35
            if valid_labels[i] == 34 and y_pred[i] == 35:
                num_same = num_same + 1
            if valid_labels[i] == 34 and y_pred[i] == 35:
                num_same = num_same + 1

            # DR2+:7  屈光介质混浊-增殖性DR可能性大:35
            if valid_labels[i] == 7 and y_pred[i] == 35:
                num_same = num_same + 1
            if valid_labels[i] == 35 and y_pred[i] == 7:
                num_same = num_same + 1

    temp_acc =num_same / len(valid_files)
    return round(temp_acc, 2)


# 现在没用到，以前kaggle DR时候学习用途
def get_train(class_num=5, test_ratio=0.1,  ):
    data1 = pd.read_csv("data1/trainLabels.csv")
    # The frac keyword argument specifies the fraction of rows to return in the random sample, so frac=1 means return all rows (in random order).
    # data1=data1.sample(frac=1)
    #random.shuffle(list_csv[i])   下面每一类打乱顺序也很好

    # list_csv = [[]] * 5   #matrix = [array] * 3操作中，只是创建3个指向array的引用
    # list_csv = [[],[],[],[],[]]

    # list_csv= [[0 for i in range(1)] for i in range(class_num)]
    list_csv = [[]*1 for i in range(class_num)]
    list_csv_train = [[] * 1 for i in range(class_num)]
    list_csv_test = [[] * 1 for i in range(class_num)]

    list_count = [0, 0, 0, 0, 0]

    test_ratio = 0.1

    for index, row in data1.iterrows():  # 获取每行的index、row
        # print(row[0],row[1])  image_file, image_class
        # print(type(row[1]))  #int

        list_csv[row[1]].append(row[0])
        list_count[row[1]] = list_count[row[1]] + 1

    for i in range(0,class_num):
        random.shuffle(list_csv[i])   #打乱顺序

        split_num=int(len(list_csv[i]) * (1-test_ratio))

        # get train data1
        for j in range(0, split_num):
            list_csv_train[i].append(list_csv[i][j])

        # get test data1
        for j in range(split_num, len(list_csv[i] )):
            list_csv_test[i].append(list_csv[i][j])

    return list_csv_train, list_csv_test

'''


def split_csv_img_seg(file_csv, file_csv_train, file_csv_valid,
                      valid_ratio=0.1, shuffle=True, random_state=2222):

    df = pd.read_csv(file_csv)

    import sklearn
    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    split_num = int(len(df) * (1 - valid_ratio))

    data_train = df[0:split_num]
    data_valid = df[split_num:]

    with open(file_csv_train, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])
        for _, row in data_train.iterrows():
            image_file = row['images']
            image_file_mask = row['masks']

            csv_writer.writerow([image_file, image_file_mask])

    with open(file_csv_valid, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])
        for _, row in data_valid.iterrows():
            image_file = row['images']
            image_file_mask = row['masks']

            csv_writer.writerow([image_file, image_file_mask])


def write_csv_img_seg(file_csv, dir):
    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(dir, False):
            for f in files:
                image_file_original = os.path.join(dir_path, f)
                (dirname, filename) = os.path.split(image_file_original)
                (filename, extension) = os.path.splitext(filename)
                if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                    continue

                if '/masks/' in image_file_original:
                    continue

                image_file_mask = image_file_original.replace('/images/', '/masks/')

                csv_writer.writerow([image_file_original, image_file_mask])

