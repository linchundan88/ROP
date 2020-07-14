
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))
import cv2
from LIBS.ImgPreprocess.my_image_helper import image_to_square

BASE_DIR = '/media/ubuntu/data2/posterior_2020_4_27/Plus_2020_02_12'
dir_original = os.path.join(BASE_DIR, 'original')
dir_preprocess = os.path.join(BASE_DIR, 'resized')
dir_preprocess384 = os.path.join(BASE_DIR, 'preprocess384')

dir_blood_vessel_seg = os.path.join(BASE_DIR, 'blood_vessel_seg_result_2020_4_27')

dir_draw_circle = os.path.join(BASE_DIR, 'draw_circle')
dir_crop_optic_disc = os.path.join(BASE_DIR, 'crop_optic_disc')
dir_optic_disc_seg = os.path.join(BASE_DIR, 'optic_disc_seg')
dir_optic_disc_error = os.path.join(BASE_DIR, 'optic_disc_error')

dir_blood_vessel_seg_posterior = os.path.join(BASE_DIR, 'blood_vessel_seg_posterior')


DO_PREPROCESS = True
DO_PREPROCESS_384 = True

DO_BLOOD_VESSEL_SEG = False

DO_OPTIC_DISC_DETECTION = True
DO_DRAW_CIRCLE = True
DO_CROP_POSTERIOR = False
CROP_POSTERIOR_CIRCLE = False

DO_BLOOD_VESSEL_SEG_POSTERIOR = True


# 640 * 512
if DO_PREPROCESS:
    from LIBS.ImgPreprocess.my_rop import resize_rop_dir
    resize_rop_dir(dir_original, dir_preprocess,
        resize_shape=(640, 480), padding_top=16, padding_bottom=16,
                   output_shape=(640, 512))

# 384 * 384
if DO_PREPROCESS_384:
    from LIBS.ImgPreprocess.my_rop import resize_rop_dir
    resize_rop_dir(dir_original, dir_preprocess384,
                   resize_shape=(640, 480),
                   image_to_square=True, output_shape=(384, 384))


if DO_BLOOD_VESSEL_SEG:
    PATCH_H = 64
    PATCH_W = 64

    dicts_models = []
    model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/ROP_013-0.968-013-0.571_0.724.hdf5'
    dict_model1 = {'model_file': model_file1,
                   'input_shape': (299, 299, 3), 'model_weight': 1}
    dicts_models.append(dict_model1)

    model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/014-0.974-014-0.551_0.707.hdf5'
    dict_model1 = {'model_file': model_file1,
                   'input_shape': (299, 299, 3), 'model_weight': 0.8}
    dicts_models.append(dict_model1)

    model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/vessel_segmentation/patch_based/ROP_008-0.971-008-0.585_0.737.hdf5'
    dict_model1 = {'model_file': model_file1,
                   'input_shape': (299, 299, 3), 'model_weight': 0.8}
    dicts_models.append(dict_model1)

    for dir_path, subpaths, files in os.walk(dir_preprocess, False):
        for f in files:
            img_file_source = os.path.join(dir_path, f)

            filename, file_extension = os.path.splitext(img_file_source)
            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            from LIBS.ImgPreprocess.my_patches_based_seg import seg_blood_vessel
            img_result = seg_blood_vessel(img_file_source, dicts_models, PATCH_H, PATCH_W,
                                          rop_resized=True, threshold=127, min_size=10, tmp_dir='/tmp',
                                          test_time_image_aug=True)

            image_file_dest = img_file_source.replace(dir_preprocess, dir_blood_vessel_seg)
            os.makedirs(os.path.dirname(image_file_dest), exist_ok=True)
            print(image_file_dest)
            cv2.imwrite(image_file_dest, img_result)


if DO_OPTIC_DISC_DETECTION:
    #region Configurations and loading model
    from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_optic_disc_config import OpticDiscConfig
    from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_seg_optic_disc import seg_optic_disc, optic_disc_draw_circle, crop_posterior
    from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib

    config = OpticDiscConfig()
    config.display()

    config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = 1   # len(image) must match BATCH_SIZE

    model = modellib.MaskRCNN(mode="inference", config=config)
    image_shape = (384, 384, 1)

    weights_path = '/home/ubuntu/dlp/deploy_models/ROP/segmentation_optic_disc/mask_rcnn_opticdisc_0022_loss0.2510.h5'
    model.load_weights(weights_path, by_name=True)
    #endregion

    for dir_path, subpaths, files in os.walk(dir_preprocess384, False):
        for f in files:
            img_file_preprocess384 = os.path.join(dir_path, f)
            img_file_source = img_file_preprocess384.replace(dir_preprocess384, dir_original)

            filename, file_extension = os.path.splitext(img_file_preprocess384)
            if file_extension.upper() not in ['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']:
                print('file ext name:', f)
                continue

            img_file_mask_tmp = img_file_preprocess384.replace(dir_preprocess384, dir_optic_disc_seg)
            (confidence, img_file_mask, circle_center, circle_diameter) = seg_optic_disc(model, img_file_preprocess384,
                            img_file_mask_tmp, image_shape=image_shape, return_optic_disc_postition=True)
            if confidence is not None:
                print('detect optic disc successfully! ', img_file_preprocess384)

                if DO_DRAW_CIRCLE:
                    img_draw_circle = optic_disc_draw_circle(img_file_preprocess384, circle_center, circle_diameter, diameter_times=3)
                    img_file_draw_circle = img_file_preprocess384.replace(dir_preprocess384, dir_draw_circle)
                    if not os.path.exists(os.path.dirname(img_file_draw_circle)):
                        os.makedirs(os.path.dirname(img_file_draw_circle))
                    cv2.imwrite(img_file_draw_circle, img_draw_circle)

                if DO_CROP_POSTERIOR:
                    img_crop_optic_disc = crop_posterior(img_file_preprocess384, circle_center, circle_diameter,
                            diameter_times=3, image_size=299, crop_circle=CROP_POSTERIOR_CIRCLE)
                    img_file_crop_optic_disc = img_file_preprocess384.replace(dir_preprocess384, dir_crop_optic_disc)
                    if not os.path.exists(os.path.dirname(img_file_crop_optic_disc)):
                        os.makedirs(os.path.dirname(img_file_crop_optic_disc))
                    cv2.imwrite(img_file_crop_optic_disc, img_crop_optic_disc)

                if DO_BLOOD_VESSEL_SEG_POSTERIOR:
                    img_blood_vessel_seg = image_to_square(img_file_preprocess384.replace(dir_preprocess384, dir_blood_vessel_seg))
                    img_blood_vessel_seg = cv2.resize(img_blood_vessel_seg, (384, 384))
                    img_blood_vessel_seg_posterior = crop_posterior(img_blood_vessel_seg, circle_center, circle_diameter,
                            diameter_times=3, image_size=299, crop_circle=CROP_POSTERIOR_CIRCLE)

                    img_file_blood_vessel_seg_posterior = img_file_preprocess384.replace(dir_preprocess384, dir_blood_vessel_seg_posterior)
                    if not os.path.exists(os.path.dirname(img_file_blood_vessel_seg_posterior)):
                        os.makedirs(os.path.dirname(img_file_blood_vessel_seg_posterior))
                    cv2.imwrite(img_file_blood_vessel_seg_posterior, img_blood_vessel_seg_posterior)

            else:
                print('detect optic disc fail! ', dir_preprocess384)
                img_file_optic_disc_error = dir_preprocess384.replace(dir_preprocess384, dir_optic_disc_error)

                if not os.path.exists(os.path.dirname(img_file_optic_disc_error)):
                    os.makedirs(os.path.dirname(img_file_optic_disc_error))
                import shutil
                shutil.copy(img_file_source, img_file_optic_disc_error)

print('OK')

