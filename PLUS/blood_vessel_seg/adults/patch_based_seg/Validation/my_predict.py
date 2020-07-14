import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import cv2

img_file = '/tmp4/img1.png'
img_file = 'rop1.jpg'
img_file = '/media/ubuntu/data2/BloodVesselsSegment_2019_10_22/original/DRIVE/training/images/24_training.tif'

PATCH_H = 64
PATCH_W = 64

import keras
model_file = '/home/ubuntu/dlp/deploy_models/vessel_segmentation/transfer_vessel_seg_patch-012-0.968_0.68_0.81.hdf5'
model1 = keras.models.load_model(model_file, compile=False)

image1 = cv2.imread(img_file)

from LIBS.ImgPreprocess.my_patches_based_seg import seg_blood_vessel
img_result = seg_blood_vessel(image1, model1, PATCH_H, PATCH_W,
            threshold=127, min_size=10, tmp_dir='/tmp',
                            test_time_image_aug=True)

cv2.imwrite('a2.png', img_result)

pass
