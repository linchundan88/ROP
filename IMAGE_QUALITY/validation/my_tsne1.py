import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from LIBS.DataPreprocess.my_data import get_images_labels

save_tsne_image = "/tmp5/t_sne_2020_5_21/Image_quality/Ungradable1.png"
nb_classes = 2
filename_csv = os.path.join(os.path.abspath('..'),
                'datafiles/dataset4', 'image_quality_split_patid_test.csv')
df = pd.read_csv(filename_csv)
files, labels = get_images_labels(filename_csv_or_pd=df)

#region compute probs
# model_dir = '/home/ubuntu/dlp/deploy_models/ROP/image_quality/2020_2_24'
# model_file = os.path.join(model_dir, 'InceptionV3-009-0.953.hdf5')
# model_file = os.path.join(model_dir, 'Xception-010-0.957.hdf5')
# model_file = os.path.join(model_dir, 'NASNetMobile-008-0.959.hdf5')
# input_shape = (299, 299, 3)

model_file = '/home/ubuntu/dlp/deploy_models/ROP/image_quality/2020_5_19/NASNetMobile-008-0.965.hdf5'
input_shape = (224, 224, 3)

from LIBS.TSNE.my_tsne_helper import compute_features, gen_tse_features, draw_tsne
features = compute_features(model_file, files, input_shape=input_shape)
X_tsne = gen_tse_features(features)
save_npy_file = "/tmp5/probs_test_image_quality.npy"
import numpy as np
np.save(save_npy_file, X_tsne)
# X_tsne = np.load(save_npy_file)

draw_tsne(X_tsne, labels, nb_classes=nb_classes, save_tsne_image=save_tsne_image,
    labels_text=['Gradable', 'Ungradable'], colors=['g', 'r'])

print('OK')