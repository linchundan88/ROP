'''
  RPC Service
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
from xmlrpc.server import SimpleXMLRPCServer
import keras
import uuid
import cv2

PATCH_H = 64
PATCH_W = 64

import my_config
dir_tmp = os.path.join(my_config.dir_tmp, 'rop_blood_vessel_seg')

def server_seg_blood_vessel(image1, preprocess=True):
    from LIBS.ImgPreprocess.my_patches_based_seg import seg_blood_vessel
    img_result = seg_blood_vessel(image1, dicts_models, PATCH_H, PATCH_W,
            rop_resized=preprocess, threshold=127, min_size=10, tmp_dir='/tmp',
            test_time_image_aug=my_config.blood_vessel_seg_test_time_image_aug)

    str_uuid = str(uuid.uuid1())
    save_filename = os.path.join(dir_tmp, str_uuid + '.jpg')
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    cv2.imwrite(save_filename, img_result)

    return save_filename

#command parameters: predict class type no and port number
if len(sys.argv) == 2:  # sys.argv[0]  exe file itself
    port = int(sys.argv[1])
else:
    port = 5010


#region define models
model_dir = my_config.dir_deploy_models
dicts_models = []
model_file1 = os.path.join(model_dir, 'vessel_segmentation/patch_based/ROP_013-0.968-013-0.571_0.724.hdf5')
dict_model1 = {'model_file': model_file1,
               'input_shape': (299, 299, 3), 'model_weight': 1}
dicts_models.append(dict_model1)

# model_file1 = os.path.join(model_dir, 'vessel_segmentation/patch_based/014-0.974-014-0.551_0.707.hdf5')
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 0.2}
# dicts_models.append(dict_model1)

# model_file1 = os.path.join(model_dir, 'vessel_segmentation/patch_based/ROP_008-0.971-008-0.585_0.737.hdf5')
# dict_model1 = {'model_file': model_file1,
#                'input_shape': (299, 299, 3), 'model_weight': 0.3}
# dicts_models.append(dict_model1)

for dict_model in dicts_models:
    print('prepare to load model:', dict_model['model_file'])
    dict_model['model'] = keras.models.load_model(dict_model['model_file'], compile=False)
    print('load model:', dict_model['model_file'], ' complete')

#region test mode
if my_config.debug_mode:
    import time
    img_source = '/tmp4/rop1.jpg'
    #one second on 171 server.
    if os.path.exists(img_source):
        for i in range(3):
            print('start:',time.time())
            file_seg = server_seg_blood_vessel(img_source)
            print('end:', time.time())
    else:
        print('file:', img_source, ' does not exist.')
#endregion


server = SimpleXMLRPCServer(("localhost", port))
print("Listening on port: ", str(port))
server.register_function(server_seg_blood_vessel, "server_seg_blood_vessel")
server.serve_forever()

