
import os
import sys
sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../../"))

from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_seg_optic_disc import seg_optic_disc, optic_disc_draw_circle, crop_posterior
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import uuid
from xmlrpc.server import SimpleXMLRPCServer

import my_config
dir_tmp = os.path.join(my_config.dir_tmp, 'rop_optic_disc_seg')
from OpticDiscDetection.Mask_RCNN.optic_disc.helper.my_optic_disc_config import OpticDiscConfig
from OpticDiscDetection.Mask_RCNN.mrcnn import model as modellib
DIR_MODELS = my_config.dir_deploy_models

#region Configurations and loading model
config = OpticDiscConfig()
# config.display()

config.IMAGES_PER_GPU = 1
config.BATCH_SIZE = 1   # len(image) must match BATCH_SIZE

model = modellib.MaskRCNN(mode="inference", config=config)
image_shape = (384, 384, 1)

# weights_path = '/home/ubuntu/dlp/deploy_models/ROP/segmentation_optic_disc/mask_rcnn_opticdisc_0022_loss0.2510.h5'
weights_path = os.path.join(DIR_MODELS, 'segmentation_optic_disc', 'mask_rcnn_opticdisc_0022_loss0.2510.h5')
model.load_weights(weights_path, by_name=True)
#endregion


def detect_optic_disc(img_file_source, preprocess=False, img_file_blood_seg=None):
    str_uuid = str(uuid.uuid1())
    os.makedirs(os.path.join(dir_tmp, str_uuid), exist_ok=True)

    if preprocess:
        img_file_preprocess = os.path.join(dir_tmp, str_uuid, 'preproces384.jpg')

        # from LIBS.ImgPreprocess.my_preprocess_dir import preprocess_image
        # preprocess_image(img_file_source, img_file_preprocess,
        #      crop_size=384, is_rop=False, add_black_pixel_ratio=0.07)

        from LIBS.ImgPreprocess.my_rop import resize_rop_image
        img_preprocess = resize_rop_image(img_file_source, image_to_square=True,
                output_shape=(384, 384))
        cv2.imwrite(img_file_preprocess, img_preprocess)
    else:
        img_file_preprocess = img_file_source

    #mask rcnn (384,384,3)
    img_file_mask_tmp = os.path.join(dir_tmp, str_uuid, 'mask.jpg')
    (confidence, img_file_mask, circle_center, circle_diameter) = \
        seg_optic_disc(model, img_file_preprocess,
            img_file_mask_tmp, image_shape=image_shape, return_optic_disc_postition=True)

    if confidence is not None:
        img_file_draw_circle = os.path.join(dir_tmp, str_uuid, 'draw_circle.jpg')
        img_file_crop_optic_disc = os.path.join(dir_tmp, str_uuid, 'crop_optic_dsc.jpg')

        img_draw_circle = optic_disc_draw_circle(img_file_preprocess, circle_center, circle_diameter, diameter_times=3)
        cv2.imwrite(img_file_draw_circle, img_draw_circle)

        img_crop_optic_disc = crop_posterior(img_file_preprocess, circle_center, circle_diameter,
                diameter_times=my_config.posterior_diameter_times, image_size=299, crop_circle=False)
        cv2.imwrite(img_file_crop_optic_disc, img_crop_optic_disc)

        if img_file_blood_seg is None:
            return float(confidence[0]), img_file_mask_tmp, img_file_crop_optic_disc,\
                   img_file_draw_circle
        else:
            img_file_blood_vessel_seg_posterior = os.path.join(dir_tmp, str_uuid, 'blood_vessel_seg_posterior.jpg')
            from LIBS.ImgPreprocess.my_image_helper import image_to_square

            img_tmp1 = image_to_square(img_file_blood_seg, image_size=384)
            img_blood_vessel_seg_posterior = crop_posterior(img_tmp1, circle_center, circle_diameter,
                    diameter_times=my_config.posterior_diameter_times, image_size=299, crop_circle=False)
            cv2.imwrite(img_file_blood_vessel_seg_posterior, img_blood_vessel_seg_posterior)

            return float(confidence[0]), img_file_mask_tmp, img_file_crop_optic_disc, \
                   img_file_draw_circle, img_file_blood_vessel_seg_posterior

    else:
        if img_file_blood_seg is None:
            return None, None, None, None
        else:
            return None, None, None, None, None


# command prarmeter  only port number (no type as RPC_server_single_class.py)
if len(sys.argv) != 2:  # sys.argv[0]  exe file itself
    port = 5011
else:
    port = int(sys.argv[1])

if my_config.debug_mode:
    '''
    import time
    img_file_source = '/tmp1/00ed40ec-681c-42c3-afca-ca906ff11ecb.1.jpg'
    for i in range(3):
        # about one second in 171 server
        print('start:', time.time())
        confidence, img_file_mask_tmp, img_file_crop_optic_disc, img_file_draw_circle = \
            detect_optic_disc_mask(img_file_source, True)

        print('end:', time.time())
    '''

    img_file_source = '/media/ubuntu/data2/posterior_2020_4_27/add_posterior/original/0aac4dc7cd8eccc36b4396ad8ff65bfd1604f675#0f6c2bdb-0490-4714-879b-4ab1a33db917.13.jpg'
    img_file_blood_vessel_seg = '/media/ubuntu/data2/posterior_2020_4_27/add_posterior/blood_vessel_seg_result1/0aac4dc7cd8eccc36b4396ad8ff65bfd1604f675#0f6c2bdb-0490-4714-879b-4ab1a33db917.13.jpg'

    confidence, img_file_mask_tmp, img_file_crop_optic_disc, img_file_draw_circle, img_file_blood_vessel_seg_posterior = \
            detect_optic_disc(img_file_source, True, img_file_blood_vessel_seg)


server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
server.register_function(detect_optic_disc, "detect_optic_disc")
server.serve_forever()

print('OK')

