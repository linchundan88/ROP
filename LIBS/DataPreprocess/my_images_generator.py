'''
 train
    class my_images_generator
       support both multi-class and multi-labels and regression

    class my_images_weight_generator
       only support multi-class
       Dynamic Resampling, the number of classes is small.

    class my_images_weight_generator_big
       The number of classes is big, only used for big-class models(39 classes)

    class my_gen_weight_multi_labels

 Valudation or Test
    my_Generator_test
       used by my_dlp_helper.py do_predict_batch, implement test time image augmentation

 my_Generator_seg 针对语义分割sigmoid，  do_binary=True输出0-1,否则255
     输入: list_images, list_masks 一维list batch
     输出
             x_train.shape: (batch, 384, 384, 3)
             y_train.shape: (batch, 384, 384, 1)

 my_Generator_seg_multiclass 针对多类语义分割 要保证一张图片的多个Masks同样的imgaug变换
    输入: list_images, list_masks list_images一维list (batch), list_masks二维list (batch, num_classes)
    输出
             x_train.shape: (batch, 384, 384, 3)
             y_train.shape: (batch, 384, 384, num_classes)

'''

import random
import numpy as np
import collections

from keras.utils import np_utils
import math
from LIBS.DICOM.my_dicom import get_npy

from LIBS.ImgPreprocess import my_image_helper
from LIBS.ImgPreprocess.my_image_norm import input_norm


def get_balance_class(files, labels, weights):
    y = np.array(labels)
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        p[y == i] = weight

    # random sampling
    random_sampling = np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                                       p=(np.array(p) / p.sum()))

    random_sampling_list = random_sampling.tolist()  # ndarray to list

    r_files = []
    r_labels = []
    for i in random_sampling_list:
        r_files.append(files[i])
        r_labels.append(labels[i])

    # 增加每一种label的样本数   比采样类权重直观
    list_num = [0 for x in range(36)]
    for label1 in r_labels:
        list_num[label1] = list_num[label1] + 1

    print(list_num)

    return r_files, r_labels

# 根据每个类的样本数，进行一个运算,生成动态采样的概率
def _get_class_weight(list_class_samples_num, file_weight_power, epoch=0):

    file_object = open(file_weight_power)
    try:
        line = file_object.readline()
        line = line.strip('\n')  # 删除换行符
        weight_power = float(line)

        print('set weight_power from file weight_power.txt')

    except Exception:
        print('read weight_power file error')
        print('set weight_power automatcally')

        dict_weight_power = collections.OrderedDict()
        dict_weight_power['0'] = 0.76
        dict_weight_power['1'] = 0.75
        dict_weight_power['2'] = 0.73
        dict_weight_power['3'] = 0.72
        dict_weight_power['4'] = 0.70
        dict_weight_power['5'] = 0.69
        dict_weight_power['6'] = 0.68
        dict_weight_power['7'] = 0.67
        dict_weight_power['8'] = 0.66
        dict_weight_power['9'] = 0.65
        dict_weight_power['10'] = 0.63
        dict_weight_power['12'] = 0.62

        for (k, v) in dict_weight_power.items():
            if epoch >= int(k):
                weight_power = v

    finally:
        file_object.close()

    '''
    获取每种类别的样本数目, 然后根据weight_power进行运算
    根据目录和dict_mapping 自动生成 class_samples_num
    '''

    list_weights = op_class_weight(list_class_samples_num, weight_power)
    weight_class = np.array(list_weights)

    print("epoch：%d, weight_power:  %f" % (epoch, weight_power))
    print('resampling ratio:', np.round(weight_class, 2))

    return weight_class

def op_class_weight(class_samples, weight_power=0.7):
    list_sample_weights = []

    max_class_samples = max(class_samples)

    for _, class_samples1 in enumerate(class_samples):
        class_samples1 = (max_class_samples ** weight_power) / (class_samples1 ** weight_power)
        list_sample_weights.append(class_samples1)

    return list_sample_weights

def smooth_labels(y, label_smoothing):
    # https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        label_smoothing: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''

    y1 = y.copy()
    y1 = y1.astype(np.float32)
    assert len(y1.shape) == 2 and 0 <= label_smoothing <= 1

    y1 *= 1 - label_smoothing
    y1 += label_smoothing / y1.shape[1]

    # y1[y1 == 1] = 1-label_smoothing + label_smoothing / num_classes
    # y1[y1 == 0] = label_smoothing / num_classes

    # np.multiply(y1, 1-label_smoothing, out=y1, casting="unsafe")

    return y1


#my_images_generator is used by multi-class, multi-label and regression
class My_images_generator():
    def __init__(self, files, labels,
                 image_shape=(299, 299, 3),  num_output=1,
                 multi_labels=False, regression=False,  batch_size=32,
                 imgaug_seq=None,  do_normalize=True, ndim='2D'):

        self.files = files
        self.labels = labels
        self.image_shape = image_shape

        self.batch_size = batch_size
        self.num_output = num_output

        self.multi_labels = multi_labels
        self.regression = regression

        self.imgaug_seq = imgaug_seq

        self.do_normalize = do_normalize
        self.ndim = ndim

    def gen(self):
        n_samples = len(self.files)

        while True:
            for i in range(math.ceil(n_samples / self.batch_size)):
                sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
                files_batch = self.files[sl]
                labels_batch = self.labels[sl]

                if self.ndim == '2D':
                    x_train = my_image_helper.load_resize_images(files_batch, self.image_shape)
                else:  #3D DICOM files have save to npy files.
                    x_train = get_npy(files_batch)

                if self.imgaug_seq is not None:
                    x_train = self.imgaug_seq.augment_images(x_train)

                x_train = np.asarray(x_train, dtype=np.float16)
                if self.do_normalize:
                    x_train = input_norm(x_train)

                if not self.regression:
                    if not self.multi_labels:
                        y_train = np.asarray(labels_batch, dtype=np.uint8)
                        y_train = np_utils.to_categorical(y_train, num_classes=self.num_output)
                    else:
                        y_train = []

                        for labels_str in labels_batch:
                            # print(labels_str)
                            labels = str(labels_str).split('_')
                            # 0_1_1 or [1,2]
                            list_labels = []
                            for _, label in enumerate(labels):
                                if label == '':
                                    continue

                                list_labels.append(int(label))

                            # print(list_labels,'\n')
                            y_train.append(list_labels)

                            # convert '4_8_28' to [4,8,28]
                            '''
                            list_labels = []
                            for label1 in labels_str.split('_'):
                                if label1 == '':
                                    continue
                                list_labels.append(int(label1))

                            # convert [1,4]  to  [0,1,0,0,1,0,0...]
                            list_labels_convert = []
                            for j in range(self.num_output):
                                if j in list_labels:
                                    list_labels_convert.append(1)
                                else:
                                    list_labels_convert.append(0)
                                    
                            y_train.append(list_labels_convert)                                    
                           '''

                        y_train = np.asarray(y_train, dtype=np.uint8)
                else:  # regression
                    y_train = np.asarray(labels_batch, dtype=np.float16)

                if self.ndim == '3D':
                    x_train = np.expand_dims(x_train, axis=-1)

                yield x_train, y_train

# my_images_weight_generator is used by only multi-class
# before every epoch -resampling_dynamic and _get_balance_class
class My_images_weight_generator():
    def __init__(self, files, labels, weight_class_start, weight_class_end, balance_ratio, label_smoothing=0,
                 image_shape=(299, 299, 3), do_normalize=True,
                 batch_size=32, num_class=1, imgaug_seq=None, ndim='2D'):
        self.train_files = files
        self.train_labels = labels
        self.image_shape = image_shape

        self.do_normalize = do_normalize

        self.weight_class_start = weight_class_start
        self.weight_class_end = weight_class_end
        self.balance_ratio = balance_ratio

        self.batch_size = batch_size
        self.num_class = num_class

        self.imgaug_seq = imgaug_seq

        self.label_smoothing = label_smoothing

        self.ndim = ndim

    def resampling_dynamic(self, weight_class_start, weight_class_end, balance_ratio, epoch):
        alpha = balance_ratio ** epoch
        class_weights = weight_class_start * alpha + weight_class_end * (1 - alpha)
        class_weights = np.around(class_weights, decimals=2)  # 保留两位小数

        print('resampling ratio:', class_weights)
        return class_weights

    def gen(self):
        n_samples = len(self.train_files)

        current_batch_num = 0
        current_epoch = 0

        while True:
            weights = self.resampling_dynamic(weight_class_start=self.weight_class_start, weight_class_end=self.weight_class_end,
                     balance_ratio=self.balance_ratio, epoch=current_epoch)

            train_files_balanced, train_labels_balanced = get_balance_class(
                    self.train_files, self.train_labels, weights=weights)

            # print('\nlabels:', train_labels_balanced)
            for i in range(math.ceil(n_samples / self.batch_size)):
                sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
                files_batch, labels_batch = train_files_balanced[sl], train_labels_balanced[sl]

                if self.ndim == '2D':
                    x_train = my_image_helper.load_resize_images(files_batch, self.image_shape)
                else:
                    x_train = get_npy(files_batch)

                if self.imgaug_seq is not None:
                    x_train = self.imgaug_seq.augment_images(x_train)

                # imgs_aug返回的x_train 的是list，每个元素(299,299,3) float32
                x_train = np.asarray(x_train, dtype=np.float16)
                if self.do_normalize:
                    x_train = input_norm(x_train)

                y_train = np.asarray(labels_batch, dtype=np.uint8)  # 64*1
                y_train = np_utils.to_categorical(y_train, num_classes=self.num_class)

                if self.label_smoothing > 0:
                    y_train = smooth_labels(y_train, self.label_smoothing)

                current_batch_num = current_batch_num + 1

                if self.ndim == '3D':
                    x_train = np.expand_dims(x_train, axis=-1)

                yield x_train, y_train

            current_epoch = current_epoch + 1


# big class, only multi-class two_stage_classification.
# the keys are _get_weight_class and  _get_weight_class
class My_images_weight_generator_bigclass():
    def __init__(self, files, labels, file_weight_power, list_class_samples_num,
                 image_shape=(299, 299, 3), do_normalize=True,
                 batch_size=64, num_class=1, imgaug_seq=None):

        self.train_files = files
        self.train_labels = labels

        self.image_shape = image_shape
        self.do_normalize = do_normalize

        self.file_weight_power = file_weight_power
        self.list_class_samples_num = list_class_samples_num  # 每个类别样本数目

        self.batch_size = batch_size
        self.num_class = num_class

        self.imgaug_seq = imgaug_seq

    def gen(self):
        n_samples = len(self.train_files)

        current_epoch = 0  # dynamic weights need current_epoch

        while True:
            # Sampling weight based on every class' sample size
            weights = _get_class_weight(self.list_class_samples_num, self.file_weight_power, current_epoch)

            # balanced dataset
            train_files_balanced, train_labels_balanced = get_balance_class(
                self.train_files, self.train_labels, weights=weights)

            print('\nlabels:', train_labels_balanced)

            for i in range(math.ceil(n_samples / self.batch_size)):
                # 数组末尾不满一个批次是否没有利用? slice超过会不理会，利用全部数据
                sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
                files_batch, labels_batch = train_files_balanced[sl], train_labels_balanced[sl]

                x_train = my_image_helper.load_resize_images(files_batch, self.image_shape)
                if self.imgaug_seq is not None:
                    x_train = self.imgaug_seq.augment_images(x_train)

                x_train = np.asarray(x_train, dtype=np.float16)
                if self.do_normalize:
                    x_train = input_norm(x_train)

                y_train = np.asarray(labels_batch, dtype=np.uint8)  # (batch,1)
                y_train = np_utils.to_categorical(y_train, num_classes=self.num_class)

                yield x_train, y_train

            current_epoch += 1


class My_gen_weight_multi_labels():
    def __init__(self, files, labels, labels_single, file_weight_power, list_class_samples_num,
                 image_shape=(299, 299), do_normalize=True,
                 batch_size=32, num_class=1, imgaug_seq=None, label_smoothing=0):

        self.train_files = files
        self.train_labels = labels
        self.train_labels_single = labels_single

        self.image_shape = image_shape
        self.do_normalize = do_normalize

        self.file_weight_power = file_weight_power
        self.list_class_samples_num = list_class_samples_num  #每个类别样本数目

        self.batch_size = batch_size
        self.num_class = num_class

        self.imgaug_seq = imgaug_seq

        self.label_smoothing = label_smoothing

    def gen(self):
        n_samples = len(self.train_files)

        current_epoch = 0  # use it to set default dynamic resampling weights

        while True:
            #region get balanced dataset using dynamic resampling based on single label(smallest class)
            classes_weights = _get_class_weight(self.list_class_samples_num, self.file_weight_power, current_epoch)
            classes_weights_total = sum(classes_weights)

            files_balanced = []
            labels_balanced = []

            current_index = 0  #index train_files, may iterate train_files more than one time.
            while len(labels_balanced) < n_samples:
                #sampling probability of this single simple
                prob_sampling = classes_weights[int(self.train_labels_single[current_index])] / classes_weights_total

                max_num = 1000000
                rand1 = random.randint(1, max_num)
                if rand1 <= max_num * prob_sampling:    # This image has been selected.
                    files_balanced.append(self.train_files[current_index])
                    labels_balanced.append(self.train_labels[current_index])

                current_index += 1
                if current_index >= n_samples:
                    current_index = 0

            print('\nlabels:', labels_balanced)
            # endregion

            for i in range(math.ceil(n_samples / self.batch_size)):
                sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
                files_batch, labels_batch = files_balanced[sl], labels_balanced[sl]

                #region generate x_train(images)
                x_train = my_image_helper.load_resize_images(files_batch, self.image_shape)
                if self.imgaug_seq is not None:
                    x_train = self.imgaug_seq.augment_images(x_train)

                x_train = np.asarray(x_train, dtype=np.float16)
                if self.do_normalize:
                    x_train = input_norm(x_train)
                # endregion

                #region generate y_train(labels)
                y_train = []

                for labels_str in labels_batch:
                    labels_str = str(labels_str)

                    # labels like '0_1_1'
                    labels = labels_str.split('_')
                    list_labels_convert = []
                    for _, label in enumerate(labels):
                        if label == '':
                            continue

                        list_labels_convert.append(int(label))

                    '''
                    for label1 in labels_str.split('_'):
                        if label1 == '':
                            continue
                        list_labels.append(int(label1))

                    # convert '4_8_28' to [4,8,28]                                     
                    #convert [1,4]  to  [0,1,0,0,1,0,0...]
                    list_labels_convert = []
                    for j in range(self.num_class):
                        if j in list_labels:
                            list_labels_convert.append(1)
                        else:
                            list_labels_convert.append(0)
                    '''

                    y_train.append(list_labels_convert)

                y_train = np.asarray(y_train, dtype=np.uint8)

                if self.label_smoothing > 0:
                    y_train = smooth_labels(y_train, self.label_smoothing)
                #endregion

                yield x_train, y_train

            current_epoch += 1


def my_Generator_test(files, image_shape=(299, 299, 3), batch_size=64,
            imgaug_seq=None, do_normalize=True):

    n_samples = len(files)

    for i in range(math.ceil(n_samples/batch_size)):
        sl = slice(i * batch_size, (i + 1) * batch_size)
        files_batch = files[sl]

        images = my_image_helper.load_resize_images(files_batch, image_shape)

        if imgaug_seq is None:
            x_test = images
        else:
            x_test = imgaug_seq.augment_images(images)

        x_test = np.asarray(x_test, dtype=np.float16)
        if do_normalize:
            x_test = input_norm(x_test)

        yield x_test


def my_Generator_test_time_aug(file, image_shape=(299, 299, 3),
                do_normalize=True, random_times=0):

    images = my_image_helper.load_resize_images(file, image_shape)

    from imgaug import augmenters as iaa
    imgaug_seq = iaa.Sequential([
        iaa.Fliplr(1),
    ])
    images1 = imgaug_seq.augment_images(images)

    imgaug_seq = iaa.Sequential([
        iaa.Flipud(1)
    ])
    images2 = imgaug_seq.augment_images(images)

    imgaug_seq = iaa.Sequential([
        iaa.Fliplr(1),
        iaa.Flipud(1)
    ])
    images3 = imgaug_seq.augment_images(images)

    x_test = images + images1 + images2 + images3

    if random_times > 0:
        imgaug_seq = iaa.Sequential([
            # iaa.CropAndPad(percent=(-0.04, 0.04)),
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Flipud(0.2),  # horizontally flip 50% of the images

            # iaa.ContrastNormalization((0.94, 1.06)),
            # sometimes1(iaa.Add((-6, 6)),
            iaa.Affine(
                scale=(0.97, 1.03),
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                # rotate=(0, 360),  # rotate by -10 to +10 degrees
            ),
        ])

        for i in range(random_times):
            images_i = imgaug_seq.augment_images(images)
            x_test += images_i

    x_test = np.asarray(x_test, dtype=np.float16)
    if do_normalize:
        x_test = input_norm(x_test)

    return x_test



if __name__ == '__main__':
   pass
