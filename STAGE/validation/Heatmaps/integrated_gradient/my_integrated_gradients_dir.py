
import sys
import os

import LIBS.ImgPreprocess.my_image_helper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from LIBS.Heatmaps.IntegratedGradient.IntegratedGradients import integrated_gradients
from LIBS.DataPreprocess import my_images_generator
from LIBS.ImgPreprocess.my_image_helper import my_gen_img_tensor
from LIBS.ImgPreprocess import my_preprocess

DO_PREPROCESS = False
GEN_CSV = True

dir_original = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/original'
dir_preprocess = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/preprocess384'
dir_dest = '/tmp4/ROP训练l图集汇总_20200423_热力图入选原始图/results/stage/integrated_gradient/InceptionResnetV2'

from LIBS.ImgPreprocess import my_preprocess_dir
if DO_PREPROCESS:
    my_preprocess_dir.do_preprocess_dir(dir_original, dir_preprocess,
        image_size=384, is_rop=False, add_black_pixel_ratio=0.07)

filename_csv = os.path.join(dir_dest, 'csv', 'predict_dir.csv')

if GEN_CSV:
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    from LIBS.DataPreprocess.my_data import write_csv_dir_nolabel
    write_csv_dir_nolabel(filename_csv, dir_preprocess)


# model_file = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_3_7/Xception-010-0.981.hdf5'
model_file = '/home/ubuntu/dlp/deploy_models/ROP/STAGE/2020_3_7/InceptionResnetV2-008-0.982.hdf5'
model1 = keras.models.load_model(model_file, compile=False)
#because .model.optimizer.get_gradients, our model use some customobject
model1.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
               , metrics=['acc'])
image_size = 299

#Wrap it with integrated_gradients.
ig = integrated_gradients(model1)

df = pd.read_csv(filename_csv)

for _, row in df.iterrows():
    image_file = row['images']
    assert dir_preprocess in image_file, 'preprocess directory error'

    preprocess = False
    if preprocess:
        img_preprocess = my_preprocess.do_preprocess(image_file, crop_size=384)
        img_input = my_gen_img_tensor(img_preprocess,
                                          image_shape=(image_size, image_size, 3))
    else:
        img_input = my_gen_img_tensor(image_file,
                                          image_shape=(image_size, image_size, 3))

    prob = model1.predict(img_input)
    class_predict = np.argmax(prob)

    if class_predict == 1:
        file_dest = image_file.replace(dir_preprocess, os.path.join(dir_dest))
        os.makedirs(os.path.dirname(file_dest), exist_ok=True)

        ig_result = ig.explain(img_input[0], outc=class_predict)
        exs = []
        exs.append(ig.explain(img_input[0], outc=class_predict))

        #plot image
        th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))

        print(file_dest)

        plt.axis('off')
        plt.imshow(exs[0][:, :, 0], cmap="seismic", vmin=-1 * th, vmax=th)
        plt.savefig(file_dest, bbox_inches='tight')
        plt.close()

print('OK')


