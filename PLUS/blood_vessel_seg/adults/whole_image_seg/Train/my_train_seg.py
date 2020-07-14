
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
GPU_NUM = 3
from keras.callbacks import ModelCheckpoint
import math, collections
from LIBS.DataPreprocess.my_data import split_dataset
from LIBS.CNN_Models.my_loss.my_metrics import *
from LIBS.DataPreprocess.my_images_generator_seg import my_Generator_seg
from LIBS.CNN_Models.my_multi_gpu import ModelMGPU

#region initial setting
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_VALID = 16

TRAIN_TYPE = 'BloodVessel384'
model_save_dir = '/tmp5/' + TRAIN_TYPE

DATFILE_TYPE = 'dataset1'
csv_file = os.path.join(os.path.abspath('..'),
                        'datafiles', DATFILE_TYPE, TRAIN_TYPE +'.csv')
train_image_files, train_mask_files, valid_image_files, valid_mask_files = \
    split_dataset(csv_file, valid_ratio=0.1, field_columns=['images', 'masks'])

img_size = 384
image_shape = (img_size, img_size, 3)
#endregion

TRAIN_TIMES = 5

for i in range(TRAIN_TIMES):

    #region data generator
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

    #endregion

    #region learning rate scheduler, checkpoint callback

    def scheduler(epoch):
        try:
            file_object = open('lr.txt')
            line = file_object.readline()
            file_object.close()
            line = line.strip('\n')
            lr_rate = float(line)
            print("set learn rate by lr.txt:  %f" % (lr_rate))

            K.set_value(model1.optimizer.lr, lr_rate)
            return K.get_value(model1.optimizer.lr)
        except Exception:
            dict_lr_rate = collections.OrderedDict()
            dict_lr_rate['0'] = 1e-3  # 0.001
            dict_lr_rate['60'] = 1e-4  # 0.0001
            dict_lr_rate['90'] = 1e-5  # 0.00001
            dict_lr_rate['130'] = 1e-6  #0.000001
            dict_lr_rate['170'] = 6e-7

            for (k, v) in dict_lr_rate.items():
                if epoch >= int(k):
                    lr_rate = v

            print("epoch：%d, set learn rate: %f according to pre-defined policy." % (epoch, lr_rate))
            K.set_value(model1.optimizer.lr, lr_rate)

        return K.get_value(model1.optimizer.lr)

    change_lr = keras.callbacks.LearningRateScheduler(scheduler)

    model_save_filepath = os.path.join(model_save_dir,
                    'traintimes_' + str(i), TRAIN_TYPE ++ '-{epoch:03d}-{val_IOU:.3f}_{val_DICE:.3f}.hdf5')
    os.makedirs(os.path.dirname(model_save_filepath), exist_ok=True)
    checkpointer = ModelCheckpoint(model_save_filepath, verbose=1,
                save_weights_only=False, save_best_only=False)

    #endregion

    #region define and compile the model

    # print('loading model...')
    # model_file = os.path.abspath(os.path.join(sys.path[0], "..",
    #         'trained_models', 'BloodVessel384-102-0.670_dice0.8021.hdf5'))
    # model1 = keras.models.load_model(model_file, compile=False)

    from LIBS.CNN_Models.segmentation.my_unet_resnet_v2 import get_unet_resnet_v2
    model1 = get_unet_resnet_v2(input_shape=(img_size, img_size, 1), dropout_ratio=0.4, init_filters=16)

    # from LIBS.CNN_Models.segmentation import my_unet
    # model1 = my_u_net.get_unet1(input_shape= (img_size, img_size, 1), init_filters=32,
    #         BN=True, transpose=True, dropout_ratio=0.2, num_classes=1)

    # model1 = my_unet.get_unet1(input_shape= (img_size, img_size, 1), init_filters=64,
    #            BN=True, transpose=True, dropout_ratio=0.2, num_classes=1)


    # from LIBS.CNN_Models.segmentation.my_nested_unet import get_Nest_Unet
    # model1 = get_Nest_Unet(input_shape=(img_size, img_size, 1), num_class=1,
    #                        nb_filter=[32, 64, 128, 256, 512], BN=True, dropout_ratio=0.2)

    # from LIBS.CNN_Models.segmentation.my_dense_unet import getDenseUNet
    # model1 = getDenseUNet(input_shape= (img_size, img_size, 1), nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.0, dropout_rate=0.2, weight_decay=1e-4,
    #                  num_class=1)

    if GPU_NUM > 1:
        print('convert base model to Multiple GPU...')
        model1 = ModelMGPU(model1, GPU_NUM)
        print('convert base model to Multiple GPU OK')

    #endregion

    model1.compile(optimizer='adam', loss=dice_coef_loss, metrics=['acc', IOU, DICE])

    model1.fit_generator(
        my_gen_train,
        steps_per_epoch=math.ceil(len(train_image_files) / BATCH_SIZE_TRAIN),
        epochs=200,
        validation_data=my_gen_valid,
        validation_steps=math.ceil(len(valid_image_files) / BATCH_SIZE_VALID),
        callbacks=[change_lr, checkpointer]
    )

    K.clear_session()

print('OK')

