
import os
import pandas as pd
import numpy as np
from LIBS.DataPreprocess import my_data
from LIBS.DataPreprocess.my_images_generator import My_images_generator, My_images_weight_generator
import keras.backend as K
import keras.optimizers
from keras.callbacks import ModelCheckpoint
from LIBS.CNN_Models import my_transfer_learning
from LIBS.CNN_Models.my_multi_gpu import ModelMGPU
from LIBS.CNN_Models.my_loss.my_metrics import sensitivity, specificity
import math
from LIBS.CNN_Models.optimization.lookahead import Lookahead
from LIBS.CNN_Models.optimization.adabound import AdaBound
import json

def train_task_one_step(model1, filename_csv_train, FILENAME_CSV_VALID, filename_csv_test=None,
                        input_shape=(299, 299, 3), imgaug_train_seq=None,
                        add_top=False, change_top=True,
                        optimizer="adam", lookahead=True,
                        epoch_finetuning=None, dict_lr_finetuning=None,
                        batch_size_train=32, batch_size_valid=64,
                        label_smoothing=0, class_weight=None,
                        weight_class_start=None, weight_class_end=None, balance_ratio=None,
                        model_save_dir='/tmp', model_name='model1',
                        gpu_num=1, verbose=1,
                        config_file_realtime='config_file_realtime.json'):


    #region read csv, split train validation set
    df = pd.read_csv(filename_csv_train)
    NUM_CLASSES = df['labels'].nunique(dropna=True)

    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(FILENAME_CSV_VALID)
    #endregion

    #region load convert and compile model
    if isinstance(model1, str):
        print('loading model...')
        model1 = keras.models.load_model(model1, compile=False)
        print('loading model complete!')

    if add_top:
        model1 = my_transfer_learning.add_top(model1, num_output=NUM_CLASSES, activation_function='SoftMax')

    if change_top:
        model1 = my_transfer_learning.convert_model_transfer(model1, clsss_num=NUM_CLASSES,
                        change_top=change_top, activation_function='SoftMax',
                        freeze_feature_extractor=False)

    model_finetuning = my_transfer_learning.convert_trainable_all(model1)

    if gpu_num > 1:
        print('convert base model to Multiple GPU...')
        model_finetuning = ModelMGPU(model_finetuning, gpu_num)
        print('convert base top model to Multiple GPU OK')

    assert optimizer in ['adam', 'SGD', 'adabound'], 'optimizer type  error'
    if optimizer == 'adam':
        op_finetuning = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if optimizer == 'SGD':
        op_finetuning = keras.optimizers.sgd(lr=1e-3, momentum=0.9, nesterov=True)
    if optimizer == 'adabound':
        op_finetuning = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, amsbound=False)

    # def custom_loss(y_true, y_pred):
    #     return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    # model_finetuning.compile(loss=custom_loss,
    #         optimizer=op_finetuning, metrics=['acc'])

    model_finetuning.compile(loss='categorical_crossentropy',
            optimizer=op_finetuning, metrics=['acc'])
            # optimizer = op_finetuning, metrics = ['acc', sensitivity, specificity])

    if lookahead:
        lookahead = Lookahead(k=5, alpha=0.5)
        lookahead.inject(model_finetuning)

    #endregion

    #region data generator
    if weight_class_start is not None:
        my_gen_train = My_images_weight_generator(files=train_files, labels=train_labels, image_shape=input_shape,
                  weight_class_start=weight_class_start, weight_class_end=weight_class_end, balance_ratio=balance_ratio,
                  num_class=NUM_CLASSES, imgaug_seq=imgaug_train_seq, batch_size=batch_size_train,
                  label_smoothing=label_smoothing)
    else:
        my_gen_train = My_images_generator(files=train_files, labels=train_labels, image_shape=input_shape,
                  num_output=NUM_CLASSES, imgaug_seq=imgaug_train_seq, batch_size=batch_size_train)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels,
                image_shape=input_shape, num_output=NUM_CLASSES, batch_size=batch_size_valid)

    if filename_csv_test is not None:
        test_files, test_labels = my_data.get_images_labels(filename_csv_test)
        my_gen_test = My_images_generator(files=test_files, labels=test_labels,
                image_shape=input_shape, num_output=NUM_CLASSES, batch_size=batch_size_valid)

    #endregion

    # region save model dir and checkpointer
    os.makedirs(model_save_dir, exist_ok=True)

    save_filepath_finetuning = os.path.join(model_save_dir, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")
    checkpointer_finetuning = ModelCheckpoint(save_filepath_finetuning,
              verbose=1, save_weights_only=False, save_best_only=False)
    # endregion

    #region computer validation confusion matrix during training
    class My_callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            try:
                with open(config_file_realtime, 'r') as json_file:
                    data = json.load(json_file)

                    if data['epoch_compute_cf_train'] == 1:
                        compute_cf_train = True
                    else:
                        compute_cf_train = False

                    if data['epoch_compute_cf_valid'] == 1:
                        compute_cf_valid = True
                    else:
                        compute_cf_valid = False

                    if data['epoch_compute_cf_test'] == 1:
                        compute_cf_test = True
                    else:
                        compute_cf_test = False
            except:
                print('read realtime helper file error!')
                compute_cf_train = True
                compute_cf_valid = True
                compute_cf_test = True

            if compute_cf_train:
                print('calculate confusion matrix of training dataset...')
                # do not use img augmentation
                my_gen_train_test = My_images_generator(files=train_files, labels=train_labels,
                                                   image_shape=input_shape, num_output=NUM_CLASSES,
                                                   batch_size=batch_size_train)

                i = 0
                for x_train, y_train in my_gen_train_test.gen():
                    probabilities = self.model.predict(x_train)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(train_files) / batch_size_train):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix

                labels = [x for x in range(0, NUM_CLASSES)]
                confusion_matrix_train = sk_confusion_matrix(train_labels, y_preds, labels=labels)

                print(confusion_matrix_train)

            if compute_cf_valid:
                print('calculate confusion matrix of validation dataset...')
                i = 0
                for x_valid, y_valid in my_gen_valid.gen():
                    probabilities = self.model.predict(x_valid)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(valid_files) / batch_size_valid):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix

                labels = [x for x in range(0, NUM_CLASSES)]
                confusion_matrix_valid = sk_confusion_matrix(valid_labels, y_preds, labels=labels)

                print(confusion_matrix_valid)

            if compute_cf_test:
                print('calculate confusion matrix of test dataset...')
                i = 0
                for x_test, y_test in my_gen_test.gen():
                    probabilities = self.model.predict(x_test)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(test_files) / batch_size_valid):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, NUM_CLASSES)]
                confusion_matrix_test = sk_confusion_matrix(test_labels, y_preds, labels=labels)

                print(confusion_matrix_test)

    my_callback = My_callback()
    #endregion

    if epoch_finetuning is None:
        if len(df) > 10000:
            epoch_finetuning = 20
        elif len(df) > 5000:
            epoch_finetuning = 25
        elif len(df) > 2000:
            epoch_finetuning = 30
        else:
            epoch_finetuning = 40

    def scheduler_finetuning(epoch):
        if optimizer == 'adabound':
            return K.get_value(model_finetuning.optimizer.lr)

        try:
            with open(config_file_realtime, 'r') as json_file:
                data = json.load(json_file)
                if data['lr_rate'] > 0:
                    lr_rate = data['lr_rate']

                    print("epoch：%d, current learn rate:  %f by realtime helper file" % (epoch, lr_rate))
                    K.set_value(model_finetuning.optimizer.lr, lr_rate)
                    return K.get_value(model_finetuning.optimizer.lr)
        except Exception:
            print('read realtime helper file error!')

        if dict_lr_finetuning is not None:
            for (k, v) in dict_lr_finetuning.items():
                if epoch >= int(k):
                    lr_rate = v

            print("epoch：%d, set  learn rate:  %f according to pre-defined policy." % (epoch, lr_rate))
            K.set_value(model_finetuning.optimizer.lr, lr_rate)

        return K.get_value(model_finetuning.optimizer.lr)

    change_lr_finetuning = keras.callbacks.LearningRateScheduler(scheduler_finetuning)

    history_finetuning = model_finetuning.fit_generator(
        my_gen_train.gen(),
        steps_per_epoch=math.ceil(len(train_files) / batch_size_train),  # number of training batch
        epochs=epoch_finetuning,
        verbose=verbose,
        validation_data=my_gen_valid.gen(),
        validation_steps=math.ceil(len(valid_files) / batch_size_valid),
        callbacks=[checkpointer_finetuning, change_lr_finetuning, my_callback],
        class_weight=class_weight
    )

    K.clear_session()  #release GPU memory


def train_task_two_steps(model1, filename_csv_train, FILENAME_CSV_VALID, filename_csv_test=None,
                         input_shape=(299, 299, 3), imgaug_train_seq=None,
                         add_top=False, change_top=True, freeze_layes_num=None,
                         optimizer="adam", lookahead=True,
                         epoch_traintop=0, epoch_finetuning=0,
                         dict_lr_traintop=None, dict_lr_finetuning=None,
                         batch_size_train=32, batch_size_valid=64,
                         label_smoothing=0, class_weight=None,
                         weight_class_start=None, weight_class_end=None, balance_ratio=None,
                         model_save_dir='/tmp', model_name='model1',
                         gpu_num=1, verbose=1,
                         config_file_realtime='config_file_realtime.json'):

    #region read csv, split train validation set
    df = pd.read_csv(filename_csv_train)
    NUM_CLASSES = df['labels'].nunique(dropna=True)

    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(FILENAME_CSV_VALID)

    #endregion

    #region load , convert  train_top model
    if isinstance(model1, str):
        print('loading model...')
        model1 = keras.models.load_model(model1, compile=False)
        print('loading model complete!')

    if add_top:
        model1 = my_transfer_learning.add_top(model1, num_output=NUM_CLASSES, activation_function='SoftMax')

    model_traintop = my_transfer_learning.convert_model_transfer(model1, clsss_num=NUM_CLASSES,
            change_top=change_top, activation_function='SoftMax',
            freeze_feature_extractor=True, freeze_layes_num=freeze_layes_num)

    if gpu_num > 1:
        print('convert base model to Multiple GPU...')
        model_traintop = ModelMGPU(model_traintop, gpu_num)
        print('convert base top model to Multiple GPU OK')

    assert optimizer in ['adam', 'SGD', 'adabound'], 'optimizer type  error'
    #endregion

    #region data generator
    if weight_class_start is not None:
        my_gen_train = My_images_weight_generator(files=train_files, labels=train_labels, image_shape=input_shape,
                  weight_class_start=weight_class_start, weight_class_end=weight_class_end, balance_ratio=balance_ratio,
                  num_class=NUM_CLASSES, imgaug_seq=imgaug_train_seq, batch_size=batch_size_train,
                  label_smoothing=label_smoothing)
    else:
        my_gen_train = My_images_generator(files=train_files, labels=train_labels, image_shape=input_shape,
                 imgaug_seq=imgaug_train_seq, num_output=NUM_CLASSES, batch_size=batch_size_train)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels,
                image_shape=input_shape, num_output=NUM_CLASSES, batch_size=batch_size_valid)

    if filename_csv_test is not None:
        test_files, test_labels = my_data.get_images_labels(filename_csv_test)
        my_gen_test = My_images_generator(files=test_files, labels=test_labels,
                image_shape=input_shape, num_output=NUM_CLASSES, batch_size=batch_size_valid)

    #endregion

    # region save model dir and checkpointer
    os.makedirs(model_save_dir, exist_ok=True)

    save_filepath_traintop = os.path.join(model_save_dir, model_name + "-traintop-{epoch:03d}-{val_acc:.3f}.hdf5")
    checkpointer_traintop = ModelCheckpoint(save_filepath_traintop,
                      verbose=1, save_weights_only=False, save_best_only=False)

    save_filepath_finetuning = os.path.join(model_save_dir, model_name + "-{epoch:03d}-{val_acc:.3f}.hdf5")
    checkpointer_finetuning = ModelCheckpoint(save_filepath_finetuning,
                      verbose=1, save_weights_only=False, save_best_only=False)

    # endregion

    #region computer validation confusion matrix
    class My_callback(keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs=None):
            try:
                with open(config_file_realtime, 'r') as json_file:
                    data = json.load(json_file)

                    if data['epoch_compute_cf_train'] == 1:
                        compute_cf_train = True
                    else:
                        compute_cf_train = False

                    if data['epoch_compute_cf_valid'] == 1:
                        compute_cf_valid = True
                    else:
                        compute_cf_valid = False

                    if data['epoch_compute_cf_test'] == 1:
                        compute_cf_test = True
                    else:
                        compute_cf_test = False
            except:
                print('read realtime helper file error!')
                compute_cf_train = True
                compute_cf_valid = True
                compute_cf_test = True

            if compute_cf_train:
                print('calculate confusion matrix of training dataset...')
                # do not use img augmentation
                my_gen_train_test = My_images_generator(files=train_files, labels=train_labels,
                                                   image_shape=input_shape, num_output=NUM_CLASSES,
                                                   batch_size=batch_size_train)

                i = 0
                for x_train, y_train in my_gen_train_test.gen():
                    probabilities = self.model.predict(x_train)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(train_files) / batch_size_train):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, NUM_CLASSES)]
                confusion_matrix_train = sk_confusion_matrix(train_labels, y_preds, labels=labels)

                print(confusion_matrix_train)

            if compute_cf_valid:
                print('calculate confusion matrix of validation dataset...')
                i = 0
                for x_valid, y_valid in my_gen_valid.gen():
                    probabilities = self.model.predict(x_valid)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(valid_files) / batch_size_valid):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, NUM_CLASSES)]
                confusion_matrix_valid = sk_confusion_matrix(valid_labels, y_preds, labels=labels)

                print(confusion_matrix_valid)

            if compute_cf_test:
                print('calculate confusion matrix of test dataset...')
                i = 0
                for x_test, y_test in my_gen_test.gen():
                    probabilities = self.model.predict(x_test)
                    if i == 0:
                        probs = probabilities
                    else:
                        probs = np.vstack((probs, probabilities))

                    i += 1
                    if i == math.ceil(len(test_files) / batch_size_valid):
                        break

                y_preds = probs.argmax(axis=-1)
                y_preds = y_preds.tolist()

                from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                labels = [x for x in range(0, NUM_CLASSES)]
                confusion_matrix_test = sk_confusion_matrix(test_labels, y_preds, labels=labels)

                print(confusion_matrix_test)

            #endregion

    my_callback = My_callback()
    #endregion

    #region train header layers
    if epoch_traintop > 0:
        if optimizer == 'adam':
            op_train_top = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if optimizer == 'SGD':
            op_train_top = keras.optimizers.sgd(lr=1e-3, momentum=0.9, nesterov=True)
        if optimizer == 'adabound':
            op_train_top = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, amsbound=False)

        model_traintop.compile(loss='categorical_crossentropy',
                               optimizer=op_train_top, metrics=['acc'])
                            # optimizer = op_train_top, metrics = ['acc', sensitivity, specificity])
        if lookahead:
            lookahead = Lookahead(k=5, alpha=0.5)
            lookahead.inject(model_traintop)

        if epoch_traintop is None:
            if len(df) > 10000:
                epoch_traintop = 5
            elif len(df) > 5000:
                epoch_traintop = 8
            elif len(df) > 2000:
                epoch_traintop = 10
            else:
                epoch_traintop = 15

        def scheduler_traintop(epoch):
            if optimizer == 'adabound':
                return K.get_value(model_traintop.optimizer.lr)

            try:
                with open(config_file_realtime, 'r') as json_file:
                    data = json.load(json_file)
                    if data['lr_rate'] > 0:
                        lr_rate = data['lr_rate']

                        print("epoch：%d, current learn rate:  %f by realtime helper file" % (epoch, lr_rate))
                        K.set_value(model_traintop.optimizer.lr, lr_rate)
                        return K.get_value(model_traintop.optimizer.lr)
            except Exception:
                print('read realtime helper file error!')

            if dict_lr_traintop is not None:
                for (k, v) in dict_lr_traintop.items():
                    if epoch >= int(k):
                        lr_rate = v

                print("epoch：%d, set  learn rate:  %f according to pre-defined policy." % (epoch, lr_rate))
                K.set_value(model_traintop.optimizer.lr, lr_rate)

            return K.get_value(model_traintop.optimizer.lr)

        change_lr_traintop = keras.callbacks.LearningRateScheduler(scheduler_traintop)

        history_top = model_traintop.fit_generator(
            my_gen_train.gen(),
            steps_per_epoch=math.ceil(len(train_files) / batch_size_train), #number of training batch
            epochs=epoch_traintop,
            verbose=verbose,
            validation_data=my_gen_valid.gen(),
            validation_steps=math.ceil(len(valid_files) / batch_size_valid),
            callbacks=[checkpointer_traintop, change_lr_traintop, my_callback],
            class_weight=class_weight)

    #endregion

    #region fine tuning all layers
    if epoch_finetuning > 0:
        model_finetuning = my_transfer_learning.convert_trainable_all(model_traintop)

        if optimizer == 'adam':
            op_finetuning = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if optimizer == 'SGD':
            op_finetuning = keras.optimizers.sgd(lr=1e-5, momentum=0.9, nesterov=True)
        if optimizer == 'adabound':
            op_finetuning = AdaBound(lr=1e-04, final_lr=0.01, gamma=1e-03, amsbound=False)

        model_finetuning.compile(loss='categorical_crossentropy',
               optimizer=op_finetuning, metrics=['acc'])
            # optimizer = op_finetuning, metrics = ['acc', sensitivity, specificity])

        if lookahead:
            lookahead = Lookahead(k=5, alpha=0.5)
            lookahead.inject(model_finetuning)

        if epoch_finetuning is None:
            if len(df) > 10000:
                epoch_finetuning = 20
            elif len(df) > 5000:
                epoch_finetuning = 25
            elif len(df) > 2000:
                epoch_finetuning = 30
            else:
                epoch_finetuning = 40

        def scheduler_finetuning(epoch):
            if optimizer == 'adabound':
                return K.get_value(model_finetuning.optimizer.lr)

            try:
                with open(config_file_realtime, 'r') as json_file:
                    data = json.load(json_file)
                    if data['lr_rate'] > 0:
                        lr_rate = data['lr_rate']

                        print("epoch：%d, current learn rate:  %f by realtime helper file" % (epoch, lr_rate))
                        K.set_value(model_finetuning.optimizer.lr, lr_rate)
                        return K.get_value(model_finetuning.optimizer.lr)
            except Exception:
                print('read realtime helper file error!')

            if dict_lr_finetuning is not None:
                for (k, v) in dict_lr_finetuning.items():
                    if epoch >= int(k):
                        lr_rate = v

                print("epoch：%d, set  learn rate:  %f according to pre-defined policy." % (epoch, lr_rate))
                K.set_value(model_finetuning.optimizer.lr, lr_rate)

            return K.get_value(model_finetuning.optimizer.lr)

        change_lr_finetuning = keras.callbacks.LearningRateScheduler(scheduler_finetuning)

        history_finetuning = model_finetuning.fit_generator(
            my_gen_train.gen(),
            steps_per_epoch=math.ceil(len(train_files) / batch_size_train),  # number of training batch
            epochs=epoch_finetuning,
            verbose=verbose,
            validation_data=my_gen_valid.gen(),
            validation_steps=math.ceil(len(valid_files) / batch_size_valid),
            callbacks=[checkpointer_finetuning, change_lr_finetuning, my_callback],
            class_weight=class_weight
        )

    #endregion

    K.clear_session()  #release GPU memory


