import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from LIBS.DataPreprocess.my_data import get_images_labels

save_tsne_image = "/tmp5/t_sne_2020_5_21/Posterior_tsne.png"
nb_classes = 2
filename_csv = os.path.join(os.path.abspath('..'),
                'datafiles/dataset9', 'Posterior_split_patid_test.csv')
df = pd.read_csv(filename_csv)
files, labels = get_images_labels(filename_csv_or_pd=df)

#region compute probs
# model_file = '/home/ubuntu/dlp/deploy_models/ROP/posterior/2020_2_24/Xception-010-0.963.hdf5'
model_file ='/home/ubuntu/dlp/deploy_models/ROP/posterior/2020_6_17/Xception-007-0.951.hdf5'
input_shape = (299, 299, 3)

# model_file ='/home/ubuntu/dlp/deploy_models/ROP/posterior/2020_6_17/NASNetMobile-006-0.954.hdf5'
# input_shape = (224, 224, 3)


from LIBS.TSNE.my_tsne_helper import compute_features, gen_tse_features, draw_tsne
features = compute_features(model_file, files, input_shape=input_shape)
X_tsne = gen_tse_features(features)
save_npy_file = "/tmp5/feature_posterior_xception.npy"
import numpy as np
np.save(save_npy_file, X_tsne)
# X_tsne = np.load(save_npy_file)

draw_tsne(X_tsne, labels, nb_classes=nb_classes, save_tsne_image=save_tsne_image,
          labels_text=['Posterior', 'Non-posterior'], colors=['g', 'r'])

print('OK')