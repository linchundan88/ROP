import os
import cv2
import numpy as np
from LIBS.ImgPreprocess.my_image_helper import resize_images_dir
import csv

dir1 = '/media/ubuntu/data1/公开数据集/BloodVesselsSegment/original/IOSTAR/images'

def op_OD_masks():
    for dir_path, subpaths, files in os.walk(dir1, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)

            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            print(img_file_source)
            # 'STAR 01_OSC.jpg' 'STAR 02_ODC_ODMask.tif'
            img_file_mask_OD = img_file_source.replace('/images/', '/mask_OD/')
            img_file_mask_OD = img_file_mask_OD.replace('.jpg', '_ODMask.tif')

            img1 = cv2.imread(img_file_source)
            img2 = cv2.imread(img_file_mask_OD)

            # ret, img3 = cv2.threshold(img1, 50, 255, cv2.THRESH_BINARY)

            img1 = cv2.imread(img_file_source)
            img2 = cv2.imread(img_file_mask_OD)

            img3 = img2 // 255
            img3 = np.multiply(img1, img3)

            img_file_dest = img_file_source.replace('/original/', '/preprocess/')
            img_file_dest = img_file_dest.replace('.jpg', '.tif')
            os.makedirs(os.path.dirname(img_file_dest), exist_ok=True)
            cv2.imwrite(img_file_dest, img3)

            # 'STAR 06_ODN_GT.tif
            img_file_mask = img_file_source.replace('/images/', '/GT/')
            img_file_mask = img_file_mask.replace('.jpg', '_GT.tif')
            img1 = cv2.imread(img_file_mask)
            img3 = img2 // 255
            img3 = np.multiply(img1, img3)

            img_file_mask_dest = img_file_dest.replace('/images/', '/masks/')
            os.makedirs(os.path.dirname(img_file_mask_dest), exist_ok=True)
            cv2.imwrite(img_file_mask_dest, img3)

def resize_dir():
    resize_images_dir(source_dir='/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess/IOSTAR/',
                    dest_dir='/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess384/IOSTAR/',
                      image_size=384)

    resize_images_dir(source_dir='/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess/IOSTAR/',
                    dest_dir='/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess512/IOSTAR/',
                      image_size=512)

    resize_images_dir(source_dir='/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess/IOSTAR/',
                    dest_dir='/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess600/IOSTAR/',
                      image_size=600)

file_csv = 'IOSTAR.csv'
def gen_csv():
    with open(file_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        dir_path = '/media/ubuntu/data1/公开数据集/BloodVesselsSegment/preprocess384/IOSTAR/images'

        for dir_path, subpaths, files in os.walk(dir_path, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)

                filename, file_extension = os.path.splitext(img_file_source)

                if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                    print('file ext name:', f)
                    continue

                img_file_mask = img_file_source.replace('/images/', '/masks/')

                if os.path.exists(img_file_mask):
                    csv_writer.writerow([img_file_source, img_file_mask])

op_OD_masks()
resize_dir()
gen_csv()

print('OK')

