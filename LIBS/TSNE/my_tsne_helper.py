import os
import keras
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalAveragePooling3D, AveragePooling3D, Flatten
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# colors: b--blue, c--cyan, g--green, k--black, r--red, w--white, y--yellow, m--magenta
def draw_tsne(X_tsne, labels, nb_classes, labels_text, colors=['g', 'r', 'b'], save_tsne_image=None):

    y = np.array(labels)
    colors_map = y

    plt.figure(figsize=(10, 10))
    for cl in range(nb_classes):
        indices = np.where(colors_map == cl)
        # plt.ylabel('aaaaaaaaa')
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=colors[cl], label=labels_text[cl])
    plt.legend()

    if not os.path.exists(os.path.dirname(save_tsne_image)):
        os.makedirs(os.path.dirname(save_tsne_image))
    plt.savefig(save_tsne_image)
    # plt.show()


def compute_features(model_file, files, input_shape, batch_size=32, gen_tsne_features=True):

    model1 = keras.models.load_model(model_file, compile=False)

    for i in range(len(model1.layers) - 1, -1, -1):
        if isinstance(model1.layers[i], GlobalAveragePooling2D) or \
                isinstance(model1.layers[i], GlobalAveragePooling3D):
            layer_num_GAP = i
            break
        if isinstance(model1.layers[i], AveragePooling2D) or \
                isinstance(model1.layers[i], AveragePooling3D):
            layer_num_GAP = i
            is_avgpool = True
            break

    # Inception-Resnet V2 1536
    output_model = Model(inputs=model1.input,
                         outputs=model1.layers[layer_num_GAP].output)
    # output_model.summary()

    from LIBS.DataPreprocess import my_images_generator
    batch_no = 0
    for x in my_images_generator.my_Generator_test(files,
                image_shape=input_shape, batch_size=batch_size):

        y = output_model.predict_on_batch(x)

        print('batch_no:', batch_no)
        batch_no += 1

        if 'features' not in locals().keys():
            features = y
        else:
            features = np.vstack((features, y))

    return features


def gen_tse_features(features):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(features)

    return X_tsne