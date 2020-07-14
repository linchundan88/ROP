import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2

import keras
model_file = '/home/ubuntu/dlp/deploy_models/vessel_segmentation/transfer_vessel_seg_patch-012-0.968_0.68_0.81.hdf5'
model1 = keras.models.load_model(model_file, compile=False)

PATCH_H = 64
PATCH_W = 64

# dir_source = '/media/ubuntu/data2/STAGE/original/'
# dir_dest = '/media/ubuntu/data2/STAGE/vessel_seg/'

dir_source = '/media/ubuntu/data2/ROP_vessel_seg/PLUS血管自动分割_20191014/'
dir_dest = '/media/ubuntu/data2/ROP_vessel_seg/PLUS血管自动分割_20191014_my_results/'

for dir_path, subpaths, files in os.walk(dir_source, False):
    for f in files:
        image_file = os.path.join(dir_path, f)

        (filepath, tempfilename) = os.path.split(image_file)
        (filename, extension) = os.path.splitext(tempfilename)
        if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
            continue

        from LIBS.ImgPreprocess.my_patches_based_seg import seg_blood_vessel
        img_result = seg_blood_vessel(image_file, model1, PATCH_H, PATCH_W,
                                      rop_resized=False, test_time_image_aug=True)


        image_file_dest = image_file.replace(dir_source, dir_dest)
        os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
        print(image_file_dest)
        cv2.imwrite(image_file_dest, img_result)

print('ok')
