import os
import cv2
import numpy as np
import LIBS.ImgPreprocess.my_image_helper

def resize_rop_image(image1, padding_top=16, padding_bottom=16,
                    resize_shape=(640, 480),
                    image_to_square=False, output_shape=(640, 512)):
    if isinstance(image1, str):
        image1 = cv2.imread(image1)

    if image1.shape[0:2] != resize_shape:
        image1 = cv2.resize(image1, resize_shape)

    # img_height= image1.shape[0]
    img_width = image1.shape[1]

    image_padding_top = np.zeros((padding_top, img_width, 3), dtype=np.uint8)
    image_padding_bottom = np.zeros((padding_bottom, img_width, 3), dtype=np.uint8)
    image_resized = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)

    if image_to_square:
        image_resized = LIBS.ImgPreprocess.my_image_helper.image_to_square(image_resized)

    if image_resized.shape[0:2] != output_shape:
        image_resized = cv2.resize(image_resized, output_shape)

    return image_resized

def resize_rop_dir(base_dir, dest_dir,
                   resize_shape=(640, 480), padding_top=16, padding_bottom=16,
                   image_to_square=False, output_shape=(640, 512)):
    if not base_dir.endswith('/'):
        base_dir += '/'
    if not dest_dir.endswith('/'):
        dest_dir += '/'

    for dir_path, subpaths, files in os.walk(base_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)
            file_dir, filename = os.path.split(image_file_source)
            file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue

            image_resized = resize_rop_image(image_file_source,
                    resize_shape=resize_shape,
                    padding_top=padding_top, padding_bottom=padding_bottom,
                    image_to_square=image_to_square,
                    output_shape=output_shape)

            image_file_dest = image_file_source.replace(base_dir, dest_dir)
            os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
            cv2.imwrite(image_file_dest, image_resized)
            print(image_file_dest)