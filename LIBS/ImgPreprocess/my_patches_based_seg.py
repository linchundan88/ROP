'''
# or tf.extract_image_patches(images,ksizes, strides,rates,padding,name=None)

image:输入图像的tesnsor，必须是[batch, in_rows, in_cols, depth]类型
ksize:滑动窗口的大小，长度必须大于四

strides:每块patch区域之间中心点之间的距离，必须是: [1, stride_rows, stride_cols, 1].

rates:在原始图像的一块patch中，隔多少像素点，取一个有效像素点，必须是[1, rate_rows, rate_cols, 1]

padding:有两个取值，“VALID”或者“SAME”，“VALID”表示所取的patch区域必须完全包含在原始图像中."SAME"表示

可以取超出原始图像的部分，这一部分进行0填充。

tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 5, 5, 1],
    rates=[1, 1, 1, 1], padding='VALID')
'''

import os
import keras
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import numpy as np
from LIBS.ImgPreprocess.my_image_norm import input_norm

from LIBS.ImgPreprocess.my_rop import resize_rop_image
import cv2


def seg_blood_vessel(image1, dicts_models, patch_h, patch_w,
                     threshold=127, min_size=10, tmp_dir='/tmp',
                     rop_resized=False, test_time_image_aug=True):
    for dict_model in dicts_models:
        if ('model' not in dict_model) or (dict_model['model'] is None):
            print('prepare to load model:', dict_model['model_file'])
            dict_model['model'] = keras.models.load_model(dict_model['model_file'], compile=False)
            print('load model:', dict_model['model_file'], ' complete')

    if rop_resized: #(height,width)=(480,640)
        image1 = resize_rop_image(image1)
    elif isinstance(image1, str):
        image1 = cv2.imread(image1)

    img_height, img_width = image1.shape[:-1]

    if not test_time_image_aug:
        x_test = gen_patch_data(image1, patch_h=patch_h, patch_w=patch_w)

        model_total_weight = 0
        for dict_model in dicts_models:
            model_total_weight = dict_model['model_weight']
            if 'y_test_final' not in locals().keys():
                y_test = dict_model['model_weight'] * dict_model['model'].predict(x_test)
            else:
                y_test += dict_model['model_weight'] * dict_model['model'].predict(x_test)
        y_test /= model_total_weight

        img_result = reconstruct_patches_results(y_test, img_height, img_width, tmp_dir)
        img_final_result = seg_result_postprocess(img_result, threshold=threshold, min_size=min_size)

    else:
        from LIBS.ImgPreprocess.my_test_time_img_aug import img_move, img_flip

        x_test1 = gen_patch_data(image1, patch_h=patch_h, patch_w=patch_w)
        for dict_model in dicts_models:
            model_total_weight = dict_model['model_weight']
            if 'y_test1' not in locals().keys():
                y_test1 = dict_model['model_weight'] * dict_model['model'].predict(x_test1)
            else:
                y_test1 += dict_model['model_weight'] * dict_model['model'].predict(x_test1)
        y_test1 /= model_total_weight
        img_result1 = reconstruct_patches_results(y_test1, img_height, img_width)
        # img_final_result1 = seg_result_postprocess(img_result1, threshold=threshold, min_size=10)
        # cv2.imwrite('/tmp1/a1.jpg', img_final_result1)

        image2 = img_move(image1, dx=6, dy=6) # move dx=6, dy=6 horizonal  flip
        image2 = img_flip(image2, flip_direction=1)
        x_test2 = gen_patch_data(image2, patch_h=patch_h, patch_w=patch_w)
        for dict_model in dicts_models:
            model_total_weight = dict_model['model_weight']
            if 'y_test2' not in locals().keys():
                y_test2 = dict_model['model_weight'] * dict_model['model'].predict(x_test2)
            else:
                y_test2 += dict_model['model_weight'] * dict_model['model'].predict(x_test2)
        y_test2 /= model_total_weight
        img_result2 = reconstruct_patches_results(y_test2, img_height, img_width)
        img_result2 = img_flip(img_result2, flip_direction=1)
        img_result2 = img_move(img_result2, dx=-6, dy=-6)
        img_final_result2 = seg_result_postprocess(img_result2, threshold=threshold, min_size=min_size)
        # cv2.imwrite('/tmp1/a2.jpg', img_final_result2)

        image3 = img_move(image1, dx=-6, dy=-6)
        image3 = img_flip(image3, flip_direction=0) #vertical  flip
        x_test3 = gen_patch_data(image3, patch_h=patch_h, patch_w=patch_w)
        for dict_model in dicts_models:
            model_total_weight = dict_model['model_weight']
            if 'y_test3' not in locals().keys():
                y_test3 = dict_model['model_weight'] * dict_model['model'].predict(x_test3)
            else:
                y_test3 += dict_model['model_weight'] * dict_model['model'].predict(x_test3)
        y_test3 /= model_total_weight
        img_result3 = reconstruct_patches_results(y_test3, img_height, img_width)
        img_result3 = img_flip(img_result3, flip_direction=0)
        img_result3 = img_move(img_result3, dx=6, dy=6)
        img_final_result3 = seg_result_postprocess(img_result3, threshold=threshold, min_size=min_size)
        # cv2.imwrite('/tmp1/a3.jpg', img_final_result3)

        img_result = (img_result1 + img_result2 + img_result3) / 3
        #seg_result_postprocess

        img_final_result = seg_result_postprocess(img_result, threshold=threshold, min_size=min_size)
        # cv2.imwrite('/tmp1/a_average.jpg', img_final_result)

    return img_final_result


def extract_patch_non_overlap(full_imgs, patch_h=64, patch_w=64):
    if isinstance(full_imgs, str):
        full_imgs = cv2.imread(full_imgs)

    if full_imgs is None:
        print('error')

    img_h, img_w = full_imgs.shape[:-1]

    list1 = []

    # cv2.imwrite('full.jpg', full_imgs)
    for x in range(img_w // patch_w):
        for y in range(img_h // patch_h):
            # img2 = full_imgs[x * patch_w: (x+1) * patch_w, y * patch_h: (y+1) * patch_h, :]
            img1 = full_imgs[y * patch_h: (y+1) * patch_h, x * patch_w: (x+1) * patch_w, :]

            list1.append(img1)

            # filename = "/tmp4/h{}_w{}.jpg".format(y, x)
            # print(filename)
            # cv2.imwrite(filename, img1)

    return list1


def gen_patch_data(img_file, patch_h=64, patch_w=64):

    if isinstance(img_file, str):
        image1 = cv2.imread(img_file)
    else:
        image1 = img_file

    list_patches = extract_patch_non_overlap(image1, patch_h=patch_h, patch_w=patch_w)

    list_tmp = []
    for patch1 in list_patches:
        patch1 = patch1[:, :, 1]  # G channel
        patch1 = np.expand_dims(patch1, axis=-1)
        patch1 = np.asarray(patch1, dtype=np.float16)
        patch1 = input_norm(patch1)

        list_tmp.append(patch1)

    x_valid = np.array(list_tmp)

    return x_valid


def reconstruct_patches_results(y, img_height, img_width, tmp_dir = '/tmp'):
    y *= 255

    patch_h, patch_w = y[0].shape[:-1]
    row_num = img_height // patch_h
    # col_num = img_width // patch_w

    img1 = np.zeros((img_height, img_width, 3))

    for index in range(y.shape[0]):
        img_patch = y[index]
        filename = os.path.join(tmp_dir, '{}.jpg'.format(index))
        cv2.imwrite(filename, img_patch)

        col = index // row_num
        row = index % row_num

        img1[row * patch_h:(row + 1) * patch_h,
        col * patch_w:(col + 1) * patch_w, :] = img_patch

    return img1


def seg_result_postprocess(img1, threshold=127, min_size=10):
    ret, binary = cv2.threshold(img1, threshold, 255, cv2.THRESH_BINARY)

    if min_size == 0:
        return binary
    else:
        binary = binary.astype(np.uint8)
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary[:, :, 0], connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 50

        # your answer image
        img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255

        return img2