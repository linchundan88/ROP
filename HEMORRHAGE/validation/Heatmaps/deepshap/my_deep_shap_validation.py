'''

'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import LIBS.ImgPreprocess.my_image_helper
from LIBS.Heatmaps.deepshap.my_helper_deepshap import My_deepshap
import pandas as pd
from keras.layers import *
from LIBS.ImgPreprocess import my_preprocess
import shutil

REFERENCE_FILE = os.path.join(os.path.abspath('.'), 'ref_rop.npy')
NUM_REFERENCE = 24
DIR_SAVE_TMP = '/tmp/deepshap'

DIR_SAVE_RESULTS = '/tmp5/heatmap_2020_5_22/Hemorrhage/Deepshap/InceptionResnetV2/no_blend'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset1/preprocess384/'

blend_original_image = False

model_dir = '/home/ubuntu/dlp/deploy_models/ROP/hemorrhage/2020_3_7'
dicts_models = []
#xception batch_size:6, inception-v3 batch_size:24, InceptionResnetV2 batch_size:12
# dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-007-0.993.hdf5'),
#                'input_shape': (299, 299, 3), 'batch_size': 12}
# dicts_models.append(dict_model1)
dict_model1 = {'model_file': os.path.join(model_dir, 'Xception-005-0.989.hdf5'),
               'input_shape': (299, 299, 3), 'batch_size': 8}
dicts_models.append(dict_model1)
# dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionV3-008-0.991.hdf5'),
#                'input_shape': (299, 299, 3),  'batch_size': 24}
# dicts_models.append(dict_model1)

my_deepshap = My_deepshap(dicts_models, reference_file=REFERENCE_FILE, num_reference=NUM_REFERENCE)

#region generate heatmaps

MODEL_NO = 0
image_shape = dicts_models[MODEL_NO]['input_shape']

for predict_type_name in ['Hemorrhage_split_patid_train', 'Hemorrhage_split_patid_valid', 'Hemorrhage_split_patid_test']:
    save_dir = os.path.join(DIR_SAVE_RESULTS, predict_type_name)
    DATFILE_TYPE = 'dataset9'
    filename_csv = os.path.join(os.path.abspath('../../../'),
                                'datafiles', DATFILE_TYPE, predict_type_name + '.csv')

    df = pd.read_csv(filename_csv)
    for _, row in df.iterrows():
        image_file = row['images']
        image_label = int(row['labels'])

        preprocess = False
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                                             image_shape=image_shape)
        else:
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                                             image_shape=image_shape)

        prob = dicts_models[MODEL_NO]['model'].predict(img_input)
        class_predict = np.argmax(prob)

        if (class_predict == 1 and image_label == 1) or\
                (class_predict == 1 and image_label == 0):
            list_classes, list_images = my_deepshap.shap_deep_explainer(
                model_no=MODEL_NO,
                num_reference=NUM_REFERENCE,
                img_input=img_input, ranked_outputs=1,
                blend_original_image=blend_original_image, norm_reverse=True,
                base_dir_save=DIR_SAVE_TMP)

            if class_predict == 1 and image_label == 1:
                file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '1_1/'))
            if class_predict == 1 and image_label == 0:
                file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir, '0_1/'))

            assert dir_preprocess not in file_dest, 'heatmap file should not overwrite preprocess image file'

            if blend_original_image:
                filename, file_ext = os.path.splitext(file_dest)
                file_dest = filename + '.gif'
            os.makedirs(os.path.dirname(file_dest), exist_ok=True)
            shutil.copy(list_images[0], file_dest)
            print(file_dest)

#endregion

print('OK')