'''
是否可以和 my_predict 的多模型预测部分合并，代码冗余
但是confusion_matrix需要每个模型不匹配的数据和每个模型的confusion_matrix
而my_predict不需要
prob_list,  pred_list,  cf_list, not_match_list

delete_cf_files  从原目录中删除 confusion matrix目录中的文件
数据调整后，删除confusion matrix目录，保留或者增加新的

'''

import os
import shutil
import pandas as pd
from LIBS.DataPreprocess import my_compute_digest


#二分类，每个文件的概率
def compute_auc_dir(all_files, prob_list, dir_keep_prob):
    models_num = len(prob_list)   #合並模型的個數

    # delete directory
    if os.path.exists(os.path.dirname(dir_keep_prob)):
        shutil.rmtree(dir_keep_prob)

    for i, filename in enumerate(all_files):

        prob_0 = 0
        for j in range(models_num): #第j个模型，第i个文件
            prob_0 = prob_0 + prob_list[j][i][0]
        prob_0 = round(prob_0 / models_num, 2)

        prob_1 = 0
        for j in range(models_num):
            prob_1 = prob_1 + prob_list[j][i][1]
        prob_1 = round(prob_1 / models_num, 2)

        filename_dir = os.path.dirname(filename)
        filename_dir_new = filename_dir.replace(dir_original, dir_keep_prob)
        filename_base_new = 'prob' + str(prob_0) + '_' + str(prob_1) + '_' +\
                            os.path.basename(filename)

        filename_new = os.path.join(filename_dir_new, filename_base_new)

        if not os.path.exists(filename):
            print('file:', filename, ' not found!')
            continue
        os.makedirs(os.path.dirname(filename_new), exist_ok=True)
        shutil.copyfile(filename, filename_new)
        print('file:', filename_new, ' OK!')


def del_cf_files_by_digest(dir_cf, dir_original):
    filename_csv = '/tmp/digest.csv'
    my_compute_digest.compute_digest_dir(dir_cf, filename_csv)

    df = pd.read_csv(filename_csv)  # panda dataframe
    i = 0
    for dir_path, subpaths, files in os.walk(dir_original, False):
        for f in files:
            i += 1
            if i % 1000 == 0:
                print('i:', i)
            image_file = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            digestSha1 = my_compute_digest.CalcSha1(image_file)

            df_search = df.loc[df['digest'].isin([digestSha1])]
            if len(df_search) == 0:
                pass
            else:
                # if ('=Fundus-All-add/1.' in image_file)\
                #       or ('=Fundus-All-add/7.' in image_file) :

                print('delete file:', image_file)
                os.remove(image_file)


#confusion matrix 目录 去掉0_9 等之后，就可以和原来目录，文件匹配
def del_cf_files_by_filename(dir_cf, dir_original):
    for dir_path, subpaths, files in os.walk(dir_cf, False):
        for f in files:
            image_file = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            image_file = image_file.replace(dir_cf+'/', '')
            image_file = image_file.replace(dir_cf, '')

            list_sub_dir = image_file.split('/')
            # confusion matrix 目录 去掉0_9 等之后，就可以和原来目录，文件匹配
            list_sub_dir = list_sub_dir[1:]
            filename_temp = '/'.join(list_sub_dir)
            image_file = os.path.join(dir_cf, filename_temp)

            image_file_del = image_file.replace(dir_cf, dir_original)
            if os.path.exists(image_file_del):
                print('delete file:', image_file_del)
                os.remove(image_file_del)

#简单根据位置匹配，不适用
#confusion matrix 目录 和原来目录，文件匹配
def del_cf_files_by_filename1(dir_cf, dir_original):
    for dir_path, subpaths, files in os.walk(dir_cf, False):
        for f in files:
            image_file = os.path.join(dir_path, f)

            file_base, file_ext = os.path.splitext(image_file)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            image_file_del = image_file.replace(dir_cf, dir_original)
            if os.path.exists(image_file_del):
                print('delete file:', image_file_del)
                os.remove(image_file_del)


if __name__ == '__main__':
    exit(0)
    # 获得需要删除的目录和文件
    dir_cf = '/home/jsiec/valid_0_3_confusion_matrix'
    dir_cf_obtain = '/home/jsiec/valid_0_3_confusion_matrix-保留'
    del_cf_files_by_filename1(dir_cf_obtain, dir_cf)

    # 根据confusion matrix要删除的文件 删除原始文件  删除 0_3 子目录
    dir_original = '/home/jsiec/disk2/pics_new_2018_04_29/=Fundus-All-add'
    del_cf_files_by_filename(dir_cf, dir_original)

    # 删除预处理后的文件（不用删除预处理目录，加快）
    dir_preprocess_original = '/home/jsiec/pics/Fundus_All_pre/299'
    del_cf_files_by_filename(dir_cf, dir_preprocess_original)

    print('OK')