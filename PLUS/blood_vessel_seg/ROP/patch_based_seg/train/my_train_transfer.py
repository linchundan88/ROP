'''
compared with training from scratch, IOU improve 2%
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_NUM = 1
import math
import collections
import keras
import pandas as pd
from LIBS.DataPreprocess.my_images_generator_seg import my_Generator_seg

PATCH_SIZE = 64
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32

#region read data files
DATFILE_TYPE = 'dataset7'
csv_file_train = os.path.join(os.path.abspath('..'),
        'datafiles', DATFILE_TYPE, 'BloodVessel_patches_train.csv')
csv_file_valid = os.path.join(os.path.abspath('..'),
        'datafiles', DATFILE_TYPE, 'BloodVessel_patches_valid.csv')

df = pd.read_csv(csv_file_train)
train_image_files = df['images']
train_mask_files = df['masks']
train_image_files = train_image_files.tolist()
train_mask_files = train_mask_files.tolist()

df = pd.read_csv(csv_file_valid)
valid_image_files = df['images']
valid_mask_files = df['masks']
valid_image_files = valid_image_files.tolist()
valid_mask_files = valid_mask_files.tolist()
#endregion

model_file_transfer_base = 'transfer_vessel_seg_patch-012-0.968_0.68_0.81.hdf5'
# model_file_transfer_base = 'transfer_vessel_seg_patch-014-0.970-014-0.682_0.810.hdf5'
# model_file_transfer_base = 'transfer_014-0.969-014-0.669_0.801.hdf5'  # a little bad performance


# region data generator
image_shape = (PATCH_SIZE, PATCH_SIZE, 3)

from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.96, aug)
imgaug_train_seq = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images

    sometimes(iaa.ContrastNormalization((0.9, 1.1))),
    iaa.Sometimes(0.9, iaa.Add((-6, 6))),

])

my_gen_train = my_Generator_seg(train_image_files, train_mask_files,
                                image_shape=image_shape, batch_size=BATCH_SIZE_TRAIN,
                                imgaug_seq=imgaug_train_seq, single_channel_no=1)

my_gen_valid = my_Generator_seg(valid_image_files, valid_mask_files,
                                image_shape=image_shape, batch_size=BATCH_SIZE_VALID,
                                single_channel_no=1)

# endregion

# region model checkpoint handlers
EPOCH_FINETUNING = 15
dict_lr = collections.OrderedDict()
dict_lr['0'] = 3e-4
dict_lr['2'] = 1e-4
dict_lr['2'] = 3e-5
dict_lr['4'] = 1e-5
dict_lr['5'] = 3e-6
dict_lr['8'] = 1e-6  # 0.000001
dict_lr['12'] = 6e-7

def scheduler(epoch):
    try:
        file_object = open('lr.txt')
        line = file_object.readline()
        file_object.close()
        line = line.strip('\n')  # 删除换行符
        lr_rate = float(line)
        print("current epoch %d set learn rate:  %f by lr.txt" % (epoch, lr_rate))

        K.set_value(model1.optimizer.lr, lr_rate)
        return K.get_value(model1.optimizer.lr)
    except Exception:
        if dict_lr is not None:
            for (k, v) in dict_lr.items():
                if epoch >= int(k):
                    lr_rate = v

            print("current epoch %d learn rate: %f according to pre-defined policy." % (epoch, lr_rate))
            K.set_value(model1.optimizer.lr, lr_rate)

    return K.get_value(model1.optimizer.lr)

change_lr = keras.callbacks.LearningRateScheduler(scheduler)

# endregion

TRAIN_TIMES = 5
for i in range(TRAIN_TIMES):
    # region define and compile model
    print('loading model...')
    model1 = keras.models.load_model(model_file_transfer_base, compile=False)
    print('loading model complete!')

    if GPU_NUM > 1:
        print('convert base model to Multiple GPU...')
        from LIBS.CNN_Models.my_multi_gpu import ModelMGPU
        model1 = ModelMGPU(model1, GPU_NUM)
        print('convert base model to Multiple GPU OK')

    from LIBS.CNN_Models.my_loss.my_metrics import *
    model1.compile(optimizer='adam', loss=dice_coef_loss, metrics=['acc', IOU, DICE])
    # model1.compile(optimizer='adam', loss=combined_bce_dice_loss(w_ce=1, w_dice=4), metrics=['acc', IOU, DICE])

    # lookahead bad performance
    # from LIBS.CNN_Models.optimization.lookahead import Lookahead
    # lookahead = Lookahead(k=5, alpha=0.5)
    # lookahead.inject(model1)
    # endregion

    model_save_dir = '/tmp5/blood_segmentation_rop_2020_4_13_3'
    # model_save_filepath = os.path.join(model_save_dir, '-{epoch:03d}-{dice_coef:.3f}.hdf5')
    model_save_filepath = os.path.join(model_save_dir, 'Traintimes_' + str(i),
                                       '{epoch:03d}-{val_acc:.3f}-{epoch:03d}-{val_IOU:.3f}_{val_DICE:.3f}.hdf5')
    os.makedirs(os.path.dirname(model_save_filepath), exist_ok=True)
    checkpointer = keras.callbacks.ModelCheckpoint(model_save_filepath, verbose=1,
                                                   save_weights_only=False, save_best_only=False)

    train_history = model1.fit_generator(
        my_gen_train,
        steps_per_epoch=math.ceil(len(train_image_files) / BATCH_SIZE_TRAIN),
        epochs=EPOCH_FINETUNING,
        validation_data=my_gen_valid,
        validation_steps=math.ceil(len(valid_image_files) / BATCH_SIZE_VALID),
        callbacks=[change_lr, checkpointer]
    )

    K.clear_session()

print('OK')


