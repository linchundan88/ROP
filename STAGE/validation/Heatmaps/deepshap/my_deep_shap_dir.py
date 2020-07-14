'''

'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

DO_PREPROCESS = False
GEN_CSV = True

dir_original = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/original'
dir_preprocess = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/preprocess384'
dir_dest = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/results/stage'

from LIBS.ImgPreprocess import my_preprocess_dir
if DO_PREPROCESS:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
        image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

filename_csv = os.path.join(dir_dest, 'csv', 'stage_predict_dir.csv')
if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel
    write_csv_dir_nolabel(filename_csv, dir_preprocess)

blend_original_image = False

model_dir = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_3_7'
dicts_models = []
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-008-0.982.hdf5'),
               'input_shape': (299, 299, 3), 'batch_size': 12}
dicts_models.append(dict_model1)
# dict_model1 = {'model_file': os.path.join(model_dir, 'Xception-010-0.981.hdf5'),
#                'input_shape': (299, 299, 3), 'batch_size': 8}
# dicts_models.append(dict_model1)
# dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionV3-010-0.979.hdf5'),
#                'input_shape': (299, 299, 3), 'batch_size': 24}
# dicts_models.append(dict_model1)

my_deepshap = My_deepshap(dicts_models, reference_file=REFERENCE_FILE, num_reference=NUM_REFERENCE)

#region generate heatmaps

MODEL_NO = 0
image_shape = dicts_models[MODEL_NO]['input_shape']

df = pd.read_csv(filename_csv)
for _, row in df.iterrows():
    image_file = row['images']
    assert dir_preprocess in image_file, 'preprocess directory error'

    # region predict label
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
    #endregion

    if class_predict == 1:
        list_classes, list_images = my_deepshap.shap_deep_explainer(
            model_no=MODEL_NO,
            num_reference=NUM_REFERENCE,
            img_input=img_input, ranked_outputs=1,
            blend_original_image=blend_original_image, norm_reverse=True,
            base_dir_save='/tmp')

        file_dest = image_file.replace(dir_preprocess, os.path.join(dir_dest, 'deepshap'))

        if blend_original_image:
            filename, file_ext = os.path.splitext(file_dest)
            file_dest = filename + '.gif'

        os.makedirs(os.path.dirname(file_dest), exist_ok=True)
        print(file_dest)
        shutil.copy(list_images[0], file_dest)

#endregion

print('OK')