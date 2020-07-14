
import os
import sys
from LIBS.ImgPreprocess.my_image_seg import gen_csv_rop_patches, gen_patches_basedon_dir
from LIBS.ImgPreprocess.my_rop import resize_rop_dir
from LIBS.ImgPreprocess.my_labelme import convert_json_mask

JSON_TO_MASK = False
DO_PREPROCESS = False
GEN_PATCHES = False
GEN_CSV = True

BASE_DIR = '/media/ubuntu/data1/ROP项目/ROP_blood_vessel_seg'

for data_type in ['train', 'valid']:
    dir_original = os.path.join(BASE_DIR, data_type, 'original')
    dir_preprocess = os.path.join(BASE_DIR, data_type, 'preprocess')
    dir_patches = os.path.join(BASE_DIR, data_type, 'patches')

    if JSON_TO_MASK:
        convert_json_mask(dir_original, dir_original)

    # (640,480)->(640,512), (1600,1200)->(640,512)
    if DO_PREPROCESS:
        resize_rop_dir(dir_original, dir_preprocess)

    if GEN_PATCHES:
        if data_type == 'train':
            gen_patches_basedon_dir(dir_source=dir_preprocess, dir_dest=dir_patches,
                                    patch_w=64, patch_h=64,
                                    patch_ordered=True,
                                    patch_random=True, num_random_patch=100, random_state=99999999
                                    )
        else:
            gen_patches_basedon_dir(dir_source=dir_preprocess, dir_dest=dir_patches,
                                    patch_w=64, patch_h=64,
                                    patch_ordered=True,
                                    patch_random=False
                                    )

    if GEN_CSV:
        filename_csv = os.path.abspath(os.path.join(sys.path[0], "..",
                        'datafiles/dataset7/BloodVessel_patches_'+ data_type +'.csv'))
        gen_csv_rop_patches(filename_csv, dir=dir_patches)

print('OK!')

''' patch image aug
import numpy as np
import cv2
def imgaug_images():
    from imgaug import augmenters as iaa
    imgaug_train_seq = iaa.Sequential([
        # iaa.CropAndPad(percent=(-0.04, 0.04)),
        iaa.Affine(
            scale=(0.98, 1.02),
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            rotate=(-20, 20),  # rotate by -20 to +20 degrees
        )
    ])

    aug_times = 8

    source_dir = '/media/ubuntu/data2/BloodVesselsSegment_2019_10_22/resized/image/'

    for dir_path, subpaths, files in os.walk(source_dir, False):
        for f in files:
            image_file = os.path.join(dir_path, f)

            (filepath, tempfilename) = os.path.split(image_file)
            # 取文件后缀
            (filename, extension) = os.path.splitext(tempfilename)
            if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                continue

            img_source = cv2.imread(image_file)
            img_source = np.expand_dims(img_source, axis=0)

            image_mask_file = image_file.replace('/resized/image/', '/resized/mask/')
            img_mask = cv2.imread(image_mask_file)
            img_mask = np.expand_dims(img_mask, axis=0)

            for index in range(aug_times):
                seq_det = imgaug_train_seq.to_deterministic()
                img_source_aug = seq_det.augment_images(img_source)[0]

                image_file_aug = image_file.replace('/resized/image/', '/imgaug/image/')
                image_file_aug = image_file_aug.replace('.png', '')
                image_file_aug = image_file_aug + '_' + str(index) + '.png'

                os.makedirs(os.path.dirname(image_file_aug), exist_ok=True)

                print(image_file_aug)
                cv2.imwrite(image_file_aug, img_source_aug)


                img_mask_aug = seq_det.augment_images(img_mask)[0]

                image_file_mask_aug = image_mask_file.replace('/resized/mask/', '/imgaug/mask/')
                image_file_mask_aug = image_file_mask_aug.replace('.png', '')
                image_file_mask_aug = image_file_mask_aug + '_' + str(index) + '.png'

                os.makedirs(os.path.dirname(image_file_mask_aug), exist_ok=True)
                print(image_file_aug)
                cv2.imwrite(image_file_mask_aug, img_mask_aug)

            if not '/HRF/' in image_file:
                pass

    pass

# imgaug_images()
'''

