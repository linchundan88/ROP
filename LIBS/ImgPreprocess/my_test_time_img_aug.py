# test time image augmentation,

import cv2
import numpy as np

def img_move(img1, dx, dy):
    if isinstance(img1, str):
        img2 = cv2.imread(img1)
    else:
        img2 = img1

    rows, cols = img2.shape[0:2]

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(img2, M, (cols, rows))

    return dst

def img_flip(img1, flip_direction=1): #horizonal:1, vertical:0
    if isinstance(img1, str):
        img2 = cv2.imread(img1)
    else:
        img2 = img1

    img3 = cv2.flip(img2, flip_direction)

    return img3


if __name__ == '__main__':
    img_file = '/tmp1/ouzel1.jpg'

    img1 = cv2.imread(img_file)

    image2=cv2.flip(img1,1)  #horizonal
    # image=cv2.flip(image,0)  #vertical

    cv2.imshow('aaa', image2)

    # cv2.imshow( 'dst', img_move(img_file, 50,3))
    cv2.waitKey(0)


