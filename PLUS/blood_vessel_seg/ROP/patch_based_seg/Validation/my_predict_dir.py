import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import cv2
from LIBS.ImgPreprocess.my_image_helper import image_to_square


# dir_original = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12/original'
# dir_preprocess = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12/resized_2020_4_27'
# dir_dest = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12/blood_vessel_seg_result_2020_4_27'

dir_original = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12/original'
dir_preprocess = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12/resized'
dir_dest = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12/blood_vessel_seg_result_2020_4_27'


DO_PREPROCESS = True
if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_rop import resize_rop_dir
    # (640,480)->(640,512), (1600,1200)->(640,512)
    resize_rop_dir(dir_original, dir_preprocess)

IMAGE_TO_SQUARE = False


PATCH_H = 64
PATCH_W = 64

dicts_models = []
model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/ROP_013-0.968-013-0.571_0.724.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/014-0.974-014-0.551_0.707.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 0.2}
dicts_models.append(dict_model1)

model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/ROP_008-0.971-008-0.585_0.737.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 0.3}
dicts_models.append(dict_model1)

for dir_path, subpaths, files in os.walk(dir_preprocess, False):
    for f in files:
        image_file = os.path.join(dir_path, f)

        (filepath, tempfilename) = os.path.split(image_file)
        (filename, extension) = os.path.splitext(tempfilename)
        if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
            continue

        if IMAGE_TO_SQUARE:
            img_input = image_to_square(image_file)
        else:
            img_input = image_file

        from LIBS.ImgPreprocess.my_patches_based_seg import seg_blood_vessel

        img_result = seg_blood_vessel(img_input, dicts_models, PATCH_H, PATCH_W,
                                      rop_resized=True, threshold=127, min_size=10, tmp_dir='/tmp',
                                      test_time_image_aug=True)

        image_file_dest = image_file.replace(dir_preprocess, dir_dest)
        if not os.path.exists(os.path.dirname(image_file_dest)):
            os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
        print(image_file_dest)
        cv2.imwrite(image_file_dest, img_result)

print('ok')
