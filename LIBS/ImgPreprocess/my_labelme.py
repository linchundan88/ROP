import json
import os
import cv2
import numpy as np


def convert_json_mask(base_dir, dest_dir):

    for dir_path, subpaths, files in os.walk(base_dir, False):
        for f in files:
            image_file_source = os.path.join(dir_path, f)

            file_dir, filename = os.path.split(image_file_source)
            file_base, file_ext = os.path.splitext(filename)  # 分离文件名与扩展名
            if file_ext.lower() not in ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                continue
            image_file_json = image_file_source.replace('.jpg', '.json')
            if not os.path.exists(image_file_json):
                continue
            image_file_mask = image_file_source.replace(base_dir, dest_dir)
            file_base1, file_ext1 = os.path.splitext(image_file_mask)
            image_file_mask = file_base1 +'_mask' + file_ext1

            with open(image_file_json, 'r') as load_f:
                print(image_file_json)

                img_original = cv2.imread(image_file_source)
                (height, width) = img_original.shape[0:2]
                img_original = np.zeros((height, width, 1))

                load_dict = json.load(load_f)
                for shape1 in load_dict['shapes']:
                    points = shape1['points']
                    array_point = None
                    for point in points:
                        tmp_point = np.array([[int(float(point[0])), int(float(point[1]))]])
                        if array_point is None:
                            array_point = tmp_point
                        else:
                            array_point = np.concatenate((array_point, tmp_point), axis=0)

                    cv2.fillConvexPoly(img_original, array_point, (255))

                os.makedirs(os.path.dirname(image_file_mask), exist_ok=True)
                cv2.imwrite(image_file_mask, img_original)