
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
from keras.layers import *
import shutil
import pandas as pd
import LIBS.ImgPreprocess.my_image_helper
from LIBS.ImgPreprocess import my_preprocess
from LIBS.Heatmaps.my_helper_heatmaps_CAM import get_CNN_model, server_cam,\
    server_grad_cam, server_gradcam_plusplus


DO_PREPROCESS = False
GEN_CSV = True

dir_original = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/original'
dir_preprocess = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/preprocess384'
dir_dest = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/results/stage/CAM'

from LIBS.ImgPreprocess import my_preprocess_dir
if DO_PREPROCESS:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
        image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

filename_csv = os.path.join(dir_dest, 'csv', 'predict_dir.csv')

if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel
    write_csv_dir_nolabel(filename_csv, dir_preprocess)

#region load and convert models

model_dir = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_3_7'
dicts_models = []
dict_model1 = {'model_file': os.path.join(model_dir, 'InceptionResnetV2-008-0.982.hdf5'),
               'input_shape': (299, 299, 3), 'batch_size': 12}
dicts_models.append(dict_model1)
# dict_model1 = {'model_file': os.path.join(model_dir, 'Xception-010-0.981.hdf5'),
#                'input_shape': (299, 299, 3), 'batch_size': 8}
# dicts_models.append(dict_model1)

for dict1 in dicts_models:
    print('prepare to load model:' + dict1['model_file'])
    original_model, output_model, all_amp_layer_weights1 = get_CNN_model(dict1['model_file'])

    if 'input_shape' not in dict1:
        if original_model.input_shape[2] is not None:
            dict1['input_shape'] = original_model.input_shape[1:]
        else:
            dict1['input_shape'] = (299, 299, 3)

    dict1['model_original'] = original_model
    dict1['model_cam'] = output_model
    dict1['all_amp_layer_weights'] = all_amp_layer_weights1

    print('model load complete!')

#endregion

blend_original_image = True
cam_relu = False

for heatmap_type in ['CAM', 'grad_cam', 'gradcam_plus']:
    df = pd.read_csv(filename_csv)
    for _, row in df.iterrows():
        image_file = row['images']
        assert dir_preprocess in image_file, 'preprocess directory error'

        preprocess = False
        input_shape = dicts_models[0]['input_shape']
        if preprocess:
            img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(img_preprocess,
                                                image_shape=input_shape)
        else:
            img_source = image_file
            img_input = LIBS.ImgPreprocess.my_image_helper.my_gen_img_tensor(image_file,
                                                image_shape=input_shape)

        model1 = dicts_models[0]['model_original']
        probs = model1.predict(img_input)
        class_predict = np.argmax(probs)

        if class_predict == 1:
            if heatmap_type == 'grad_cam':
                filename_CAM1 = server_grad_cam(dicts_models=dicts_models, model_no=0,
                                                    img_source=img_source, pred=class_predict,
                                                    preprocess=False, blend_original_image=blend_original_image, base_dir_save='/tmp/temp_cam/')
            if heatmap_type == 'CAM':
                filename_CAM1 = server_cam(dicts_models=dicts_models, model_no=0,
                            img_source=img_source, pred=class_predict,
                            cam_relu=cam_relu, preprocess=False, blend_original_image=blend_original_image, base_dir_save='/tmp/temp_cam/')

            if heatmap_type == 'gradcam_plus':
                filename_CAM1 = server_gradcam_plusplus(dicts_models=dicts_models, model_no=0,
                            img_source=img_input, pred=class_predict,
                            preprocess=False, blend_original_image=blend_original_image,
                            base_dir_save='/tmp3/temp_cam/')

            save_dir = os.path.join(dir_dest, heatmap_type)
            file_dest = image_file.replace(dir_preprocess, os.path.join(save_dir))
            os.makedirs(os.path.dirname(file_dest), exist_ok=True)
            shutil.copy(filename_CAM1, file_dest)
            print(file_dest)

print('OK!')