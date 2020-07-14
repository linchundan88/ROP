import csv
import os
import cv2
import pandas as pd
import uuid
from sklearn.feature_extraction.image import extract_patches_2d

from LIBS.ImgPreprocess.my_patches_based_seg import extract_patch_non_overlap


def gen_csv_rop_patches(file_csv, dir):
    os.makedirs(os.path.dirname(file_csv), exist_ok=True)

    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(dir, False):
            for f in files:
                image_file_original = os.path.join(dir_path, f)
                (dirname, filename) = os.path.split(image_file_original)
                (file_basename, file_ext) = os.path.splitext(filename)
                if not file_ext.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                    continue
                if file_basename.endswith('_mask'):
                    continue

                image_file_mask = os.path.join(dirname, file_basename + '_mask' + file_ext)

                csv_writer.writerow([image_file_original, image_file_mask])


def gen_csv(file_csv, dir):
    os.makedirs(os.path.dirname(file_csv), exist_ok=True)

    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(dir, False):
            for f in files:
                image_file_original = os.path.join(dir_path, f)
                (dirname, file_basename) = os.path.split(image_file_original)
                (file_basename, file_ext) = os.path.splitext(file_basename)
                if not file_ext.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                    continue

                image_file_mask = image_file_original.replace('/image/', '/mask/')
                image_file_mask = image_file_mask.replace('/images/', '/masks/')

                csv_writer.writerow([image_file_original, image_file_mask])


def gen_patches_basedon_dir(dir_source, dir_dest,
                            patch_ordered=True, patch_w=64, patch_h=64,
                            patch_random=False, num_random_patch=100, random_state=99999):

    if dir_source.endswith('/'):
        dir_source = dir_source[:-1]
    if dir_dest.endswith('/'):
        dir_dest += dir_dest[:-1]

    for dir_path, subpaths, files in os.walk(dir_source, False):
        for f in files:
            image_file = os.path.join(dir_path, f)

            (filepath, tempfilename) = os.path.split(image_file)
            (filename, extension) = os.path.splitext(tempfilename)
            if not extension.upper() in ['.BMP', '.PNG', '.JPEG', '.JPG', '.TIFF', '.TIF']:
                continue

            file_base, file_ext = os.path.splitext(image_file)
            if file_base.endswith('_mask'):
                continue

            image_file_mask = file_base + '_mask' + file_ext
            assert os.path.exists(image_file_mask), 'mask file not found:' + image_file_mask

            #region non-overlap patch
            if patch_ordered:
                list_patch_images = extract_patch_non_overlap(image_file, patch_h, patch_w)
                for index, patch1 in enumerate(list_patch_images):
                    image_file_patch = file_base + '_order_' + str(index) + '.png'
                    image_file_patch = image_file_patch.replace(dir_source, dir_dest)
                    os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                    print(image_file_patch)
                    cv2.imwrite(image_file_patch, patch1)

                list_patch_images_mask = extract_patch_non_overlap(image_file_mask, patch_h, patch_w)
                for index, patch1 in enumerate(list_patch_images_mask):
                    image_file_patch = file_base + '_order_' + str(index) + '_mask' + '.png'
                    image_file_patch = image_file_patch.replace(dir_source, dir_dest)
                    os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                    print(image_file_patch)
                    cv2.imwrite(image_file_patch, patch1)

            #endregion

            #region random patches
            if patch_random:
                img1 = cv2.imread(image_file)
                patches1 = extract_patches_2d(img1, (patch_w, patch_h),
                                              max_patches=num_random_patch, random_state=random_state)
                for index, patch in enumerate(patches1):
                    image_file_patch = file_base + '_random_' + str(index) + '.png'
                    image_file_patch = image_file_patch.replace(dir_source, dir_dest)
                    os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                    print(image_file_patch)
                    cv2.imwrite(image_file_patch, patch)

                img2 = cv2.imread(image_file_mask)
                patches2 = extract_patches_2d(img2, (patch_w, patch_h),
                                              max_patches=num_random_patch, random_state=random_state)
                for index, patch in enumerate(patches2):
                    image_file_patch = file_base + '_random_' + str(index) + '_mask' + '.png'
                    image_file_patch = image_file_patch.replace(dir_source, dir_dest)
                    os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                    print(image_file_patch)
                    cv2.imwrite(image_file_patch, patch)

            #endregion




def gen_patches_basedon_csv(filename_csv, dir_dest,
                            patch_ordered=True, patch_w=64, patch_h=64,
                            patch_random=False, max_patch=100, random_state=99999):

    if dir_dest.endswith('/'):
        dir_dest += dir_dest[:-1]

    df = pd.read_csv(filename_csv)
    for _, row in df.iterrows():
        image_file = row['images']
        mask_file = row['masks']

        str_uuid = str(uuid.uuid1())

        # _, filename = os.path.split(image_file)
        # filename_base, file_extension = os.path.splitext(image_file)

        #region non-overlap patch
        if patch_ordered:
            list_patch_images = extract_patch_non_overlap(image_file, patch_h, patch_w)

            for index, patch1 in enumerate(list_patch_images):
                image_file_patch = os.path.join(dir_dest, str_uuid, 'order_' + str(index) + '.png')
                os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                print(image_file_patch)
                cv2.imwrite(image_file_patch, patch1)

            list_patch_images = extract_patch_non_overlap(mask_file, patch_h, patch_w)

            for index, patch1 in enumerate(list_patch_images):
                image_file_patch = os.path.join(dir_dest, str_uuid, 'order_' + str(index) + '_mask.png')
                os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                print(image_file_patch)
                cv2.imwrite(image_file_patch, patch1)

        #endregion

        #region random patches
        if patch_random:
            img1 = cv2.imread(image_file)
            patches1 = extract_patches_2d(img1, (patch_w, patch_h), max_patches=max_patch, random_state=random_state)

            for index, patch in enumerate(patches1):
                image_file_patch = os.path.join(dir_dest, str_uuid, 'random_' + str(index) + '.png')
                os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                print(image_file_patch)
                cv2.imwrite(image_file_patch, patch)

            img1 = cv2.imread(mask_file)
            patches1 = extract_patches_2d(img1, (patch_w, patch_h), max_patches=max_patch, random_state=random_state)

            for index, patch in enumerate(patches1):
                image_file_patch = os.path.join(dir_dest, str_uuid, 'random_' + str(index) + '_mask.png')
                os.makedirs(os.path.dirname(image_file_patch), exist_ok=True)
                print(image_file_patch)
                cv2.imwrite(image_file_patch, patch)

        #endregion

