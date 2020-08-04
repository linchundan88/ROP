'''
based on https://github.com/slundberg/shap
overcome the GPU memory limitation by split and concat alogrithm.
'''
import copy
import os
import uuid
import shap
from matplotlib import pylab as plt
import numpy as np
import math
import keras
import cv2
from LIBS.ImgPreprocess.my_image_norm import input_norm_reverse

class My_deepshap():

    def __init__(self, dicts_models, reference_file, num_reference):

        self.dicts_models = dicts_models
        background = np.load(reference_file)

        self.list_e = []  # model no
        for i, dict_model in enumerate(dicts_models):
            if 'model' not in dict_model:
                print('loading model {} ...'.format(i))
                dict_model['model'] = keras.models.load_model(dict_model['model_file'], compile=False)
                print('loading model {} complete!'.format(i))

            batch_size = dict_model['batch_size']
            split_times = math.ceil(num_reference / batch_size)

            list_split_e = [] # background split
            for j in range(split_times):
                print('converting model {0} batch {1} ...'.format(i, j))
                sl = slice(j * batch_size, (j + 1) * batch_size)
                background_sl = background[sl]
                e = shap.DeepExplainer(dict_model['model'], background_sl)  # it will take 10 seconds
                list_split_e.append(e)

                print('converting model {0} batch {1} completed'.format(i, j))
            self.list_e.append(list_split_e)

    def shap_deep_explainer(self, model_no,
                            num_reference, img_input, norm_reverse=True,
                            blend_original_image=False, gif_fps=1,
                            ranked_outputs=1, base_dir_save='/tmp/DeepExplain'
                            ):

        # region mini-batch because of GPU memory limitation
        list_shap_values = []

        batch_size = self.dicts_models[model_no]['batch_size']
        split_times = math.ceil(num_reference / batch_size)
        for i in range(split_times):
            #shap 0.26
            #shap 0.4, check_additivity=False
            # shap_values_tmp1 = self.list_e[model_no][i].shap_values(img_input, ranked_outputs=ranked_outputs,
            #                                 check_additivity=check_additivity)
            shap_values_tmp1 = self.list_e[model_no][i].shap_values(img_input, ranked_outputs=ranked_outputs,
                                            )

            # shap_values ranked_outputs
            # [0] [0] (1,299,299,3)
            # [1] predict_class array
            shap_values_copy = copy.deepcopy(shap_values_tmp1)
            list_shap_values.append(shap_values_copy)

        for i in range(ranked_outputs):
            for j in range(len(list_shap_values)):
                if j == 0:
                    shap_values_tmp2 = list_shap_values[0][0][i]
                else:
                    shap_values_tmp2 += list_shap_values[j][0][i]

            shap_values_results = copy.deepcopy(list_shap_values[0])
            shap_values_results[0][i] = shap_values_tmp2 / split_times

        # endregion

        # region save files
        str_uuid = str(uuid.uuid1())
        list_classes = []
        list_images = []
        for i in range(ranked_outputs):
            predict_class = int(shap_values_results[1][0][i])  # numpy int 64 - int
            list_classes.append(predict_class)

            save_filename = os.path.join(base_dir_save, str_uuid,
                                         'Shap_Deep_Explainer{}.jpg'.format(predict_class))
            os.makedirs(os.path.dirname(save_filename), exist_ok=True)
            list_images.append(save_filename)

        pred_class_num = len(shap_values_results[0])

        if blend_original_image:
            if norm_reverse:
                img_original = np.uint8(input_norm_reverse(img_input[0]))
            else:
                img_original = np.uint8(img_input[0])
            img_original_file = os.path.join(os.path.dirname(list_images[0]), 'deepshap_original.jpg')
            cv2.imwrite(img_original_file, img_original)

        for i in range(pred_class_num):
            # predict_max_class = attributions[1][0][i]
            attribution1 = shap_values_results[0][i]

            # attributions.shape: (1, 299, 299, 3)
            data = attribution1[0]
            data = np.mean(data, -1)

            abs_max = np.percentile(np.abs(data), 100)
            abs_min = abs_max

            # dx, dy = 0.05, 0.05
            # xx = np.arange(0.0, data1.shape[1], dx)
            # yy = np.arange(0.0, data1.shape[0], dy)
            # xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
            # extent = xmin, xmax, ymin, ymax

            # cmap = 'RdBu_r'
            # cmap = 'gray'
            cmap = 'seismic'
            plt.axis('off')
            # plt.imshow(data1, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
            # plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)

            # fig = plt.gcf()
            # fig.set_size_inches(2.99 / 3, 2.99 / 3)  # dpi = 300, output = 700*700 pixels
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            if blend_original_image:
                plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
                save_filename1 = list_images[i]
                plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
                plt.close()

                img_heatmap = cv2.imread(list_images[i])
                (tmp_height, tmp_width) = img_original.shape[:-1]
                img_heatmap = cv2.resize(img_heatmap, (tmp_width, tmp_height))
                img_heatmap_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.jpg'.format(i))
                cv2.imwrite(img_heatmap_file, img_heatmap)

                dst = cv2.addWeighted(img_original, 0.65, img_heatmap, 0.35, 0)
                img_blend_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_blend_{0}.jpg'.format(i))
                cv2.imwrite(img_blend_file, dst)

                # region create gif
                import imageio
                mg_paths = [img_original_file, img_heatmap_file, img_blend_file]
                gif_images = []
                for path in mg_paths:
                    gif_images.append(imageio.imread(path))
                img_file_gif = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.gif'.format(i))
                imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
                list_images[i] = img_file_gif
                # endregion
            else:
                plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
                save_filename1 = list_images[i]
                plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
                plt.close()

        # endregion

        return list_classes, list_images

