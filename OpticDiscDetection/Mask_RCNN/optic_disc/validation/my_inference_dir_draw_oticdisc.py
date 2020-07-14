
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
import cv2
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_optic_disc_config import OpticDiscConfig
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_seg_optic_disc import seg_optic_disc, optic_disc_draw_circle, crop_posterior
from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib

#region Configurations and loading model
config = OpticDiscConfig()
config.display()

config.IMAGES_PER_GPU = 1
config.BATCH_SIZE = 1   # len(image) must match BATCH_SIZE

model = modellib.MaskRCNN(mode="inference", config=config)
image_shape = (384, 384, 1)

weights_path = '/home/ubuntu/dlp/deploy_models/ROP/segmentation_optic_disc/mask_rcnn_opticdisc_0022_loss0.2510.h5'
model.load_weights(weights_path, by_name=True)
#endregion

dir_original = '/media/ubuntu/data1/ROP_dataset/Plus/2020_02_12/original'
dir_preprocess384 = '/media/ubuntu/data1/ROP_dataset/Plus/2020_02_12/preprocess384'
dir_crop_optic_disc = '/tmp5/Plus_results_2020_4_14/crop_optic_disc'
# dir_crop_optic_disc = '/tmp5/Plus_results_2020_4_14/crop_optic_disc_crop_circle'
dir_draw_circle = '/tmp5/Plus_results_2020_4_14/draw_circle'
dir_optic_disc_seg = '/tmp5/Plus_results_2020_4_14/optic_disc_seg'
dir_dest_error = '/tmp5/Plus_results_2020_4_14/optic_disc_seg_error'


CROP_CIRCLE = False

for dir_path, subpaths, files in os.walk(dir_preprocess384, False):
    for f in files:
        img_file_source = os.path.join(dir_path, f)

        filename, file_extension = os.path.splitext(img_file_source)
        if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
            print('file ext name:', f)
            continue

        img_file_mask_tmp = img_file_source.replace(dir_preprocess384, dir_optic_disc_seg)

        (confidence, img_file_mask, circle_center, circle_diameter) = seg_optic_disc(model, img_file_source,
                        img_file_mask_tmp, image_shape=image_shape, return_optic_disc_postition=True)
        if confidence is not None:
            print('detect optic disc successfully! ', img_file_source)

            img_draw_circle = optic_disc_draw_circle(img_file_source, circle_center, circle_diameter, diameter_times=3)
            img_file_draw_circle = img_file_source.replace(dir_preprocess384, dir_draw_circle)
            os.makedirs(os.path.dirname(img_file_draw_circle), exist_ok=True)
            cv2.imwrite(img_file_draw_circle, img_draw_circle)

            img_crop_optic_disc = crop_posterior(img_file_source, circle_center, circle_diameter,
                                                 diameter_times=3, image_size=299, crop_circle=CROP_CIRCLE)
            img_file_crop_optic_disc = img_file_source.replace(dir_preprocess384, dir_crop_optic_disc)
            os.makedirs(os.path.dirname(img_file_crop_optic_disc), exist_ok=True)
            cv2.imwrite(img_file_crop_optic_disc, img_crop_optic_disc)

        else:
            print('detect optic disc fail! ', img_file_source)
            img_file_original = img_file_source.replace(dir_preprocess384, dir_original)
            img_file_dest_error = img_file_source.replace(dir_preprocess384, dir_dest_error)
            #
            os.makedirs(os.path.dirname(img_file_dest_error), exist_ok=True)
            import shutil
            shutil.copy(img_file_original, img_file_dest_error)

print('OK')

#