'''
generating reference data1
modified 2020_3_19 mid night,  add input_norm(img_black)
generate new ref_rop.npy
'''

import os
import numpy as np
import pandas as pd
import sklearn.utils
from LIBS.DataPreprocess import my_images_generator

# filename_csv images after preprocessed
filename_csv = os.path.join(os.path.abspath('.'),  'Stage.csv')

df = pd.read_csv(filename_csv)
df = sklearn.utils.shuffle(df, random_state=22222)

SAMPLES_NUM = 64
ADD_BLACK_INTERVAL = 8

imagefiles = df[0:SAMPLES_NUM]['images'].tolist()

image_shape = (299, 299, 3)

my_gen_test = my_images_generator.my_Generator_test(files=imagefiles,
     image_shape=image_shape, do_normalize=True, batch_size=SAMPLES_NUM)
x_train = my_gen_test.__next__()

#add black images
img_black = np.zeros(image_shape)
from LIBS.ImgPreprocess.my_image_norm import input_norm
img_black = input_norm(img_black)
img_black = np.expand_dims(img_black, axis=0)

for i in range(SAMPLES_NUM):
    if (i % ADD_BLACK_INTERVAL == 0):
        x_train = np.append(x_train, img_black, axis=0)

# x_train = x_train[:SAMPLES_NUM]  #clip to sample_num after add black images
x_train = np.asarray(x_train, dtype=np.float16)


save_filename = 'ref_rop.npy'
np.save(save_filename, x_train)

background = np.load(save_filename)

print('OK')

