
import cv2
import os
import numpy as np
import sys
import csv
from LIBS.ImgPreprocess.my_image_seg import gen_csv, gen_patches_basedon_dir, gen_patches_basedon_csv

def convert_image_mask_name(dir_source, dir_preprocess):
    if dir_source.endswith('/'):
        dir_source = dir_source[:-1]
    if dir_preprocess.endswith('/'):
        dir_preprocess += dir_preprocess[:-1]

    for dir_path, subpaths, files in os.walk(dir_source, False):
        for f in files:
            image_file = os.path.join(dir_path, f)

            (filepath, tempfilename) = os.path.split(image_file)
            # 取文件后缀
            (filename, extension) = os.path.splitext(tempfilename)
            if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                continue

            if '/HRF/' in image_file:
                if '/images/' in image_file:
                    # images '01_dr.jpg  manual 01_g.tif
                    image_mask_file = image_file.replace('/images/', '/manual1/')
                    image_mask_file = image_mask_file.replace('.jpg', '.tif')
                    image_mask_file = image_mask_file.replace('.JPG', '.tif')

                    image_file_dest = os.path.join(dir_preprocess, 'HRF', filename+'.png')
                    image_file_mask_dest = os.path.join(dir_preprocess, 'HRF', filename+'_mask.png')
                else:
                    continue
            elif '/DRIVE/' in image_file:
                if '/images/' in image_file:
                    # 21_training.tif 01_test.tif 21_manual1.gif
                    image_mask_file = image_file.replace('/images/', '/1st_manual/')
                    image_mask_file = image_mask_file.replace('_training.tif', '_manual1.png')
                    image_mask_file = image_mask_file.replace('_test.tif', '_manual1.png')

                    image_file_dest = os.path.join(dir_preprocess, 'DRIVE', filename + '.png')
                    image_file_mask_dest = os.path.join(dir_preprocess, 'DRIVE', filename + '_mask.png')
                else:
                    continue
            elif '/ChaseDB1/' in image_file:
                if ('1stHO.png' not in image_file) and ('2ndHO.png' not in image_file):
                    # Image_01L.jpg, Image_01L_1stHO.png
                    image_mask_file = image_file.replace('.jpg', '_1stHO.png')

                    image_file_dest = os.path.join(dir_preprocess, 'ChaseDB1', filename + '.png')
                    image_file_mask_dest = os.path.join(dir_preprocess, 'ChaseDB1', filename + '_mask.png')
                else:
                    continue
            elif '/Stare/' in image_file:
                if 'stare-images' in image_file:
                    # all-images im0001.png  labels-ah im0001.ah.png
                    image_mask_file = image_file.replace('/stare-images/', '/labels-ah/')
                    image_mask_file = image_mask_file.replace('.png', '.ah.png')

                    image_file_dest = os.path.join(dir_preprocess, 'Stare', filename + '.png')
                    image_file_mask_dest = os.path.join(dir_preprocess, 'Stare', filename + '_mask.png')
                else:
                    continue
            elif '/IOSTAR/' in image_file:
                if 'images' in image_file:
                    # STAR 08_OSN.tif , STAR 01_OSC.tif
                    image_mask_file = image_file.replace('/images/', '/masks/')

                    image_file_dest = os.path.join(dir_preprocess, 'IOSTAR', filename + '.png')
                    image_file_mask_dest = os.path.join(dir_preprocess, 'IOSTAR', filename + '_mask.png')
                else:
                    continue

            else:
                continue

            os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
            os.makedirs(os.path.dirname(image_file_mask_dest), exist_ok=True)

            image1 = cv2.imread(image_file)
            cv2.imwrite(image_file_dest, image1)

            image1 = cv2.imread(image_mask_file)
            cv2.imwrite(image_file_mask_dest, image1)

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

def gen_seg_csv(file_csv, dir_source):
    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(dir_source, False):
            for f in files:
                image_file = os.path.join(dir_path, f)

                if '_mask.png' in image_file:
                    continue

                (filepath, tempfilename) = os.path.split(image_file)
                # 取文件后缀
                (filename, extension) = os.path.splitext(tempfilename)
                if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                    continue

                image_file_mask = image_file.replace('.png', '_mask.png')
                csv_writer.writerow([image_file, image_file_mask])


if __name__ == '__main__':
    # convert_image_mask_name('/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/original',
    #                         '/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/preprocess')

    # from LIBS.ImgPreprocess.my_image_helper import resize_images_dir
    # resize_images_dir('/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/preprocess',
    #     '/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/preprocess512', image_size=512)

    # resize_images_dir('/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/preprocess',
    #     '/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/preprocess384', image_size=384)


    # dir_preprocess = '/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/preprocess512'
    # csv_file = os.path.abspath(os.path.join(sys.path[0], 'BloodVessel.csv'))
    # gen_csv(csv_file, dir_preprocess)
    #
    # from LIBS.DataPreprocess.my_data import split_csv_img_seg
    # split_csv_img_seg(file_csv='BloodVessel.csv',
    #           file_csv_train='BloodVessel_train.csv',
    #           file_csv_valid='BloodVessel_valid.csv')

    dir_patches = '/media/ubuntu/data2/BloodVesselsSegment_2020_1_20/patches'

    csv_file_train = os.path.abspath(os.path.join(sys.path[0], 'BloodVessel_train.csv'))
    csv_file_valid = os.path.abspath(os.path.join(sys.path[0], 'BloodVessel_valid.csv'))

    gen_patches_basedon_csv(filename_csv=csv_file_train, dir_dest=os.path.join(dir_patches, 'train'),
                            patch_ordered=True, patch_w=64, patch_h=64,
                            patch_random=True, max_patch=100, random_state=99999999
                            )
    gen_patches_basedon_csv(filename_csv=csv_file_valid, dir_dest=os.path.join(dir_patches, 'valid'),
                            patch_ordered=True, patch_w=64, patch_h=64,
                            patch_random=False, max_patch=100, random_state=99999999
                            )

    csv_file_train_patches = os.path.abspath(os.path.join(sys.path[0], 'BloodVessel_patches_train.csv'))
    csv_file_valid_patches = os.path.abspath(os.path.join(sys.path[0], 'BloodVessel_patches_valid.csv'))

    gen_seg_csv(csv_file_train_patches, os.path.join(dir_patches, 'train'))
    gen_seg_csv(csv_file_valid_patches, os.path.join(dir_patches, 'valid'))


print('OK')