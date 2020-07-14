
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import shutil
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_seg_optic_disc import seg_optic_disc, optic_disc_draw_circle, crop_posterior
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_optic_disc_config import OpticDiscConfig
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

DIR_DEST_BASE = '/media/ubuntu/data2/tmp5/2020_3_13_results/detect_optic_disc'
dir_original = '/media/ubuntu/data1/ROP_dataset1/original'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset1/preprocess384'

dir_tmp = '/media/ubuntu/data1/tmp1'


for dir_path, subpaths, files in os.walk(dir_preprocess, False):
    for f in files:
        img_file_source = os.path.join(dir_path, f)

        file_dir, filename = os.path.split(img_file_source)
        filename_base, file_extension = os.path.splitext(filename)

        if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
            print('file ext name:', f)
            continue

        img_file_mask_tmp = img_file_source.replace(dir_preprocess, dir_tmp)

        (confidence, img_file_mask, circle_center, circle_diameter) = seg_optic_disc(model, img_file_source,
            img_file_mask=img_file_mask_tmp, image_shape=image_shape, return_optic_disc_postition=True)
        if confidence is not None:
            print('detect optic disc successfully! ', img_file_source)

            img_file_original = img_file_source.replace(dir_preprocess, dir_original)
            img_file_dest = os.path.join(DIR_DEST_BASE, '1', str(round(confidence[0], 2)) + filename)
            os.makedirs(os.path.dirname(img_file_dest), exist_ok=True)
            shutil.copy(img_file_original, img_file_dest)

        else:
            print('detect optic disc fail! ', img_file_source)

            img_file_original = img_file_source.replace(dir_preprocess, dir_original)
            img_file_dest = os.path.join(DIR_DEST_BASE, '0', filename)
            os.makedirs(os.path.dirname(img_file_dest), exist_ok=True)
            shutil.copy(img_file_original, img_file_dest)

print('OK')

