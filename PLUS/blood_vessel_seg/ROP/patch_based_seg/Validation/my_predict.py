import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2

PATCH_H = 64
PATCH_W = 64

img_file = '/tmp4/rop1.jpg'
from LIBS.ImgPreprocess.my_image_helper import image_to_square
# img1 = image_to_square(img_file)

dicts_models = []

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/ROP_013-0.968-013-0.571_0.724.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/014-0.974-014-0.551_0.707.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)

# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/ROP_008-0.971-008-0.585_0.737.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 1}
# dicts_models.append(dict_model1)

from LIBS.ImgPreprocess.my_patches_based_seg import seg_blood_vessel
# img_result = seg_blood_vessel(img1, model1, PATCH_H, PATCH_W,
#                               rop_resized=False, threshold=127, min_size=10, tmp_dir='/tmp',
#                               test_time_image_aug=True)

img_result = seg_blood_vessel(img_file, dicts_models, PATCH_H, PATCH_W,
                              rop_resized=True, threshold=127, min_size=10, tmp_dir='/tmp',
                              test_time_image_aug=True)

cv2.imwrite('/tmp5/a_new1.png', img_result)

print('OK')

