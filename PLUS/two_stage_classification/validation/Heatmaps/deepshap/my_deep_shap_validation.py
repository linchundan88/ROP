'''

'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
import LIBS.ImgPreprocess.my_image_helper
from LIBS.Heatmaps.deepshap.my_helper_deepshap import My_deepshap
import pandas as pd
from keras.layers import *
import shutil

REFERENCE_FILE = os.path.join(os.path.abspath('.'), 'ref_rop.npy')
NUM_REFERENCE = 24

DIR_SAVE_RESULTS = '/tmp5/heatmap_2020_5_23/Plus1/Deepshap/InceptionResnetV2/no_blend'
dir_preprocess = '/media/ubuntu/data1/ROP_dataset1/preprocess384/'

DIR_SAVE_TMP = '/tmp/deepshap'

blend_original_image = False

dicts_models = []
model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_5_19/InceptionResnetV2-005-0.972.hdf5'
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'batch_size': 12}
dicts_models.append(dict_model1)
# model_file1 = '/home/ubuntu/dlp/deploy_models/ROP/plus_two_stages/2020_5_19/Xception-008-0.979.hdf5'
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'batch_size': 12}
# dicts_models.append(dict_model1)

my_deepshap = My_deepshap(dicts_models, reference_file=REFERENCE_FILE, num_reference=NUM_REFERENCE)

#region generate heatmaps

MODEL_NO = 0
image_shape = dicts_models[MODEL_NO]['input_shape']

for predict_type_name in ['Plus_step_two_split_patid_train', 'Plus_step_two_split_patid_valid', 'Plus_step_two_split_patid_test']:

    filename_csv = os.path.join(os.path.abspath('../../../'),
                    'datafiles/dataset6', predict_type_name + '.csv')
    save_dir = os.path.join(DIR_SAVE_RESULTS, predict_type_name)

    df = pd.read_csv(filename_csv)
    for _, row in df.iterrows():
        image_file = row['images']
        image_label = int(row['labels'])

        img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                        image_shape=image_shape)
        prob = dicts_models[MODEL_NO]['model'].predict(img_input)
        class_predict = np.argmax(prob)


        if (class_predict == 1 and image_label == 1) or\
                (class_predict == 1 and image_label == 0):
            list_classes, list_images = my_deepshap.shap_deep_explainer(
                model_no=MODEL_NO, num_reference=NUM_REFERENCE,
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
            if not os.path.exists(os.path.dirname(file_dest)):
                os.makedirs(os.path.dirname(file_dest))
            shutil.copy(list_images[0], file_dest)
            print(file_dest)
#endregion

print('OK')