'''

 Object detection(Optic Disc) return BBOX
 Object localization(Foeva) return Center_X, Center_Y
'''

import cv2
import numpy as np


def get_boundry(img, threshold1=10, threshold2=30):
    if isinstance(img, str):
        img = cv2.imread(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    #CV2 (height,width,channel)
    (left, right, bottom, top) = (0, img.shape[1], 0, img.shape[0])

    # 所谓多维数组的阈值化处理，比如将矩阵中小于某阈值的元素全部置零。

    img = img * (img > threshold1)

    for i in range(img.shape[0] - 1):
        if np.sum(img[i, :, :]) <= threshold2 and np.sum(img[i + 1, :, :]) > threshold2:
            bottom = i
        if np.sum(img[i, :, :]) > threshold2 and np.sum(img[i + 1, :, :]) <= threshold2:
            top = i

    for i in range(img.shape[1] - 1):
        if np.sum(img[:, i, :]) <= threshold2 and np.sum(img[:, i + 1, :]) > threshold2:
            left = i
        if np.sum(img[:, i, :]) > threshold2 and np.sum(img[:, i + 1, :]) <= threshold2:
            right = i

    return left, right, bottom, top


def convert_to_center_w_h(left, right, bottom, top):
    center_x = (right + left) / 2
    center_y = (top + bottom) / 2
    width = right-left
    height = top - bottom

    return center_x, center_y, width, height

def convert_from_center_w_h(center_x, center_y, width, height):
    left = center_x - width / 2
    right = center_x + width / 2
    bottom = center_y - height / 2
    top = center_y + height / 2

    return left, right, bottom, top


if __name__ == '__main__':
    img_file1 = '/media/ubuntu/data1/公开数据集/OpticDiscDetection/Refuge/preprocess384/Validation400/masks/V0272.bmp'
    img_file1 = '/media/ubuntu/data1/公开数据集/OpticDiscDetection/Drishti-GS1_files/preprocess/masks/384/drishtiGS_101.png'
    # img_file1 = '/media/ubuntu/data1/公开数据集/OpticDiscDetection/Refuge/preprocess384/Training400/masks/Non-Glaucoma/n0253.bmp'
    (left, right, bottom, top) = get_boundry(img_file1)
    center_x, center_y, width, height = convert_to_center_w_h(left, right, bottom, top)
    print('OK')

