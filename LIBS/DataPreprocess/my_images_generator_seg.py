import numpy as np

from LIBS.ImgPreprocess.my_image_norm import input_norm
from LIBS.ImgPreprocess import my_image_helper


def my_Generator_seg(files_images, files_masks, image_shape=(299, 299, 3), do_normalize=True,
         batch_size=64, do_binary=True, imgaug_seq=None, single_channel_no=None):

    n_samples = len(files_images)

    while True:
        for i in range((n_samples + batch_size - 1) // batch_size):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            files_images_batch = files_images[sl]
            files_masks_batch = files_masks[sl]

            list_images = my_image_helper.load_resize_images(files_images_batch, image_shape)  # 训练文件列表
            list_masks = my_image_helper.load_resize_images(files_masks_batch, image_shape, grayscale=True)

            if imgaug_seq is None:
                x_train = list_images
                y_train = list_masks
            else:
                seq_det = imgaug_seq.to_deterministic()

                x_train = seq_det.augment_images(list_images)
                y_train = seq_det.augment_images(list_masks)

            x_train = np.asarray(x_train, dtype=np.float16)
            if do_normalize:
                x_train = input_norm(x_train)

            if single_channel_no is not None:
                #BGR choose green channel green 1
                x_train = x_train[:, :, :, single_channel_no]
                x_train = np.expand_dims(x_train, axis=-1)

            y_train = np.asarray(y_train, dtype=np.uint8)

            #sigmoid  经过了变换，需要二值化
            if do_binary:
                y_train //= 128 #分割，y_train是图像 分类的话不用  需要动态判断BBOX

            #返回的类型
            # x_train.shape: (batch, 384, 384, 3)  single channel: (batch, 384, 384, 1)
            #y_train.shape: (batch, 384, 384, 1)

            yield x_train, y_train


def my_Generator_seg_test(list_images, image_shape=(299, 299, 3),
                do_normalize=True, batch_size=64, single_channel_no=None):

    n_samples = len(list_images)

    while True:
        for i in range((n_samples + batch_size - 1) // batch_size):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            files_images_batch = list_images[sl]

            x_train = my_image_helper.load_resize_images(files_images_batch, image_shape)

            x_train = np.asarray(x_train, dtype=np.float16)
            if do_normalize:
                x_train = input_norm(x_train)

            if single_channel_no is not None:  #BGR choose green channel
                x_train = x_train[:, :, :, single_channel_no]
                x_train = np.expand_dims(x_train, axis=-1)

            yield x_train


import keras

#multi process support,  do not need to provide steps_per_epoch
class Generator_seg(keras.utils.Sequence):
    def __init__(self, files_images, files_masks, image_shape=(299, 299, 3),
             batch_size=64, do_binary=True, imgaug_seq=None, single_channel_no=None):

        self.files_images = files_images
        self.files_masks = files_masks
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.do_binary = do_binary
        self.image_seq = imgaug_seq
        self.sinale_channel_no = single_channel_no

    def __len__(self):
        #Number of iterations in one epoch
        return len(self.files_images) // self.batch_size

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        files_images_batch = self.files_images[sl]
        files_masks_batch = self.files_masks[sl]

        list_images = my_image_helper.load_resize_images(files_images_batch, self.image_shape)  # 训练文件列表
        list_masks = my_image_helper.load_resize_images(files_masks_batch, self.image_shape, grayscale=True)

        if self.imgaug_seq is None:
            x_train = list_images
            y_train = list_masks
        else:
            seq_det = self.imgaug_seq.to_deterministic()

            x_train = seq_det.augment_images(list_images)
            y_train = seq_det.augment_images(list_masks)

        x_train = np.asarray(x_train, dtype=np.float16)
        x_train = input_norm(x_train)

        if self.single_channel_no is not None:
            # BGR choose green channel green 1
            x_train = x_train[:, :, :, self.single_channel_no]
            x_train = np.expand_dims(x_train, axis=-1)

        y_train = np.asarray(y_train, dtype=np.uint8)

        # sigmoid  经过了变换，需要二值化
        if self.do_binary:
            y_train //= 128  # 分割，y_train是图像 分类的话不用  需要动态判断BBOX

        # 返回的类型
        # x_train.shape: (batch, 384, 384, 3)  single channel: (batch, 384, 384, 1)
        # y_train.shape: (batch, 384, 384, 1)

        yield x_train, y_train


class Generator_seg_test(keras.utils.Sequence):
    def __init__(self, files_images, image_shape=(299, 299, 3),
             batch_size=64, do_binary=True, imgaug_seq=None, single_channel_no=None):

        self.files_images = files_images
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.do_binary = do_binary
        self.image_seq = imgaug_seq
        self.sinale_channel_no = single_channel_no

    def __len__(self):
        #Number of iterations in one epoch
        return len(self.files_images) // self.batch_size

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        files_images_batch = self.files_images[sl]

        x_train = my_image_helper.load_resize_images(files_images_batch, self.image_shape)  # 训练文件列表

        x_train = np.asarray(x_train, dtype=np.float16)
        x_train = input_norm(x_train)

        if self.single_channel_no is not None:
            # BGR choose green channel green 1
            x_train = x_train[:, :, :, self.single_channel_no]
            x_train = np.expand_dims(x_train, axis=-1)

        # x_train.shape: (batch, 384, 384, 3)  single channel: (batch, 384, 384, 1)

        yield x_train
