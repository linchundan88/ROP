import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from keras.models import load_model
from LIBS.DataPreprocess.my_images_generator_seg import my_Generator_seg_test
import cv2
from LIBS.ImgPreprocess import my_preprocess
import uuid
import shutil


def dr_seg(model1, filename_source, tmp_dir='/tmp', image_size=384, single_channel_no=None):
    image_shape = (image_size, image_size)
    image1 = my_preprocess.do_preprocess(filename_source, image_size)

    filename_preprocess = os.path.join(tmp_dir, str(uuid.uuid1()) + '.jpg')
    cv2.imwrite(filename_preprocess, image1)
    img_preprocess_seg = cv2.imread(filename_preprocess)  # (384,384,3)

    train_image_files = []
    train_image_files.append(filename_preprocess)

    gen1 = my_Generator_seg_test(train_image_files, image_shape=image_shape,
                     single_channel_no=single_channel_no)
    x = gen1.__next__()

    pred_seg = model1.predict_on_batch(x)    #1,384,384,1 batch,height,width,channels
    img_pred_0_1 = pred_seg[0] > 0.5
    img_pred_0_1 = img_pred_0_1.astype(np.int8)
    img_pred_seg = img_pred_0_1 * 255

    return img_preprocess_seg, img_pred_seg


#region single file

# model_name = 'BloodVessel384-081-iou_0.646_dice0.7852.hdf5'
# model1 = load_model(model_name, compile=False)
# filename_source = '01_test.tif'
# (img_preprocess_seg, img_pred_lesion) = dr_seg(model1, filename_source)
# cv2.imwrite('333_pred.jpg', img_pred_lesion)


model_name = 'transfer_base_BloodVessel384-102-green_iou0.670_dice0.8021.hdf5'
model1 = load_model(model_name, compile=False)

filename_source = '01_test.tif'
(img_preprocess_seg, img_pred_seg) = dr_seg(model1, filename_source,
                                            single_channel_no=1)
cv2.imwrite('01_test_result.jpg', img_pred_seg)

filename_source = 'rop1.jpg'
(img_preprocess_seg, img_pred_seg) = dr_seg(model1, filename_source,
                                            single_channel_no=1)
cv2.imwrite('rop1_result.jpg', img_pred_seg)

print('OK')
#endregion

exit(0)

#region batch dir

base_dir = '/home/jsiec/disk2/pics_new_2018_04_29/=Fundus-All-add/7.DR2'
base_dir = '/tmp/image1'
dest_dir = '/home/jsiec/segment_DR2/SoftExudates'


for dir_path, subpaths, files in os.walk(base_dir, False):
    for f in files:
        filename_source = os.path.join(dir_path, f)

        (filename, extension) = os.path.splitext(f)
        if extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
            print(filename_source)

            (img_preprocess_seg, img_pred_seg) = dr_seg(filename_source, tmp_dir='/tmp/od_seg')


            temp_dir = os.path.join(dest_dir, str(uuid.uuid1()))
            os.makedirs(temp_dir, exist_ok=True)
            shutil.copy(filename_source, os.path.join(temp_dir, 'original.jpg'))
            cv2.imwrite(os.path.join(temp_dir, 'preprocess.jpg'), img_preprocess_seg)
            cv2.imwrite(os.path.join(temp_dir, 'pred_lesion.jpg'), img_pred_seg)

#endregion

print('ok')
