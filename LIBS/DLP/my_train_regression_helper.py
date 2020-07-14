
import sys, os
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

def train_task_one_step(model1,
                        filename_csv_train, FILENAME_CSV_VALID, filename_csv_test=None,
                        image_size=299, imgaug_train_seq=None, num_output=1,
                        add_top=False, change_top=True,
                        epoch_finetuning=None, dict_lr_finetuning=None,
                        batch_size_train=32, batch_size_valid=64,
                        model_save_dir='/tmp', model_name='model1',
                        compute_cf_train=False, compute_cf_valid=False, compute_cf_test=False,
                        gpu_num=1, verbose=1):

    #region read csv, split train validation set
    df = pd.read_csv(filename_csv_train) #get number of train samples, set epoch automaticaly

    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(FILENAME_CSV_VALID)
    #endregion

    #region load and convert model
    if isinstance(model1, str):
        print('loading model...')
        model1 = keras.models.load_model(model1, compile=False)
        print('loading model complete!')

    if add_top:
        model1 = my_transfer_learning.add_top(model1, num_output=num_output, activation_function='Regression')

    model1 = my_transfer_learning.convert_model_transfer(model1, clsss_num=num_output,
                        change_top=change_top, activation_function='Regression',
                        freeze_feature_extractor=False)
    if gpu_num > 1:
        print('convert base model to Multiple GPU...')
        model1 = ModelMGPU(model1, gpu_num)
        print('convert base top model to Multiple GPU OK')

    op_adam1 = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model1.compile(loss='mse', optimizer=op_adam1, metrics=['mae', 'mse'])
    from LIBS.CNN_Models.optimization.lookahead import Lookahead
    lookahead = Lookahead(k=5, alpha=0.5)
    lookahead.inject(model1)
    #endregion

    #region data generator
    image_shape = (image_size, image_size, 3)
    my_gen_train = My_images_generator(files=train_files, labels=train_labels, regression=True,
                            imgaug_seq=imgaug_train_seq,
                            image_shape=image_shape, num_output=num_output, batch_size=batch_size_train)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels, regression=True,
                image_shape=image_shape, num_output=num_output, batch_size=batch_size_valid)

    if compute_cf_test:
        test_files, test_labels = my_data.get_images_labels(filename_csv_test)
        my_gen_test = My_images_generator(files=test_files, labels=test_labels, regression=True,
                image_shape=image_shape, num_output=num_output, batch_size=batch_size_valid)

    #endregion

    # region save model dir and checkpointer
    os.makedirs(model_save_dir, exist_ok=True)
    save_filepath_finetuning = os.path.join(model_save_dir,
               model_name + "-{epoch:03d}--{val_loss:.1f}---{val_mean_absolute_error:.1f}---{mean_squared_error:.1f}-{mean_absolute_error:.1f}.hdf5")
    checkpointer_finetuning = ModelCheckpoint(save_filepath_finetuning,
                       save_weights_only=False, save_best_only=False)
    # endregion

    #region computer validation confusion matrix during training
    class My_callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if compute_cf_train:
                print('calculate confusion matrix of training dataset...')

                my_gen_train_no_imgaug = My_images_generator(files=train_files, labels=train_labels, regression=True,
                                                   imgaug_seq=None,
                                                   image_shape=image_shape, num_output=num_output,
                                                   batch_size=batch_size_train)

                i = 0
                for x_train, y_train in my_gen_train_no_imgaug.gen():
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


    my_callback = My_callback()
    #endregion

    model_finetuning = my_transfer_learning.convert_trainable_all(model1)

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
        try:
            file_object = open('lr.txt')
            line = file_object.readline()
            file_object.close()
            line = line.strip('\n') #删除换行符
            lr_rate = float(line)

            print("epoch：%d, current learn rate:  %f by lr.txt" % (epoch, lr_rate))
            K.set_value(model_finetuning.optimizer.lr, lr_rate)

        except Exception:
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
    )


    K.clear_session()  #release GPU memory


def train_task_two_steps(model1,
                         filename_csv_train, FILENAME_CSV_VALID, filename_csv_test=None,
                         image_size=299, imgaug_train_seq=None, num_output=1,
                         add_top=False, change_top=True, freeze_layes_num=None,
                         epoch_traintop=None, epoch_finetuning=None,
                         dict_lr_traintop=None, dict_lr_finetuning=None,
                         batch_size_train=32, batch_size_valid=64,
                         model_save_dir='/tmp', model_name='model1',
                         compute_cf_train=False, compute_cf_valid=False, compute_cf_test=False,
                         gpu_num=1, verbose=1):

    #region read csv, split train validation set
    df = pd.read_csv(filename_csv_train) #get number of train samples, set epoch automaticaly
    train_files, train_labels = my_data.get_images_labels(filename_csv_train, shuffle=True)
    valid_files, valid_labels = my_data.get_images_labels(FILENAME_CSV_VALID)

    #endregion

    #region load , convert and compile train_top model
    if isinstance(model1, str):
        print('loading model...')
        model1 = keras.models.load_model(model1, compile=False)
        print('loading model complete!')

    if add_top:
        model1 = my_transfer_learning.add_top(model1, num_output=num_output, activation_function='SoftMax')

    model_traintop = my_transfer_learning.convert_model_transfer(model1, clsss_num=num_output,
            change_top=change_top, activation_function='Regression',
            freeze_feature_extractor=True, freeze_layes_num=freeze_layes_num)
    if gpu_num > 1:
        print('convert base model to Multiple GPU...')
        model_traintop = ModelMGPU(model_traintop, gpu_num)
        print('convert base top model to Multiple GPU OK')

    #endregion

    #region data generator
    image_shape = (image_size, image_size, 3)
    my_gen_train = My_images_generator(files=train_files, labels=train_labels, regression=True,
                                       imgaug_seq=imgaug_train_seq,
                                       image_shape=image_shape, num_output=num_output, batch_size=batch_size_train)

    my_gen_valid = My_images_generator(files=valid_files, labels=valid_labels, regression=True,
                image_shape=image_shape, num_output=num_output, batch_size=batch_size_valid)

    if compute_cf_test:
        test_files, test_labels = my_data.get_images_labels(filename_csv_test)
        my_gen_test = My_images_generator(files=test_files, labels=test_labels,regression=True,
                image_shape=image_shape, num_output=num_output, batch_size=batch_size_valid)

    #endregion

    # region save model dir and checkpointer
    os.makedirs(model_save_dir, exist_ok=True)

    save_filepath_traintop = os.path.join(model_save_dir,
               model_name + "-{epoch:03d}--{val_loss:.1f}---{val_mean_absolute_error:.1f}---{mean_squared_error:.1f}-{mean_absolute_error:.1f}.hdf5")
    checkpointer_traintop = ModelCheckpoint(save_filepath_traintop,
                       save_weights_only=False, save_best_only=False)

    save_filepath_finetuning = os.path.join(model_save_dir,
               model_name + "-{epoch:03d}--{val_loss:.1f}---{val_mean_absolute_error:.1f}---{mean_squared_error:.1f}-{mean_absolute_error:.1f}.hdf5")
    checkpointer_finetuning = ModelCheckpoint(save_filepath_finetuning,
                       save_weights_only=False, save_best_only=False)

    # endregion

    #region computer validation confusion matrix
    class My_callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if compute_cf_train:
                print('calculate confusion matrix of training dataset...')
                my_gen_train_no_imgaug = My_images_generator(files=train_files, labels=train_labels, regression=True,
                                                   imgaug_seq=imgaug_train_seq,
                                                   image_shape=image_shape, num_output=num_output,
                                                   batch_size=batch_size_train)

                i = 0
                for x_train, y_train in my_gen_train_no_imgaug.gen():
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


            # endregion

            if compute_cf_valid:
                print('calculate confusion matrix of validation dataset...')
                #use the same my_gen_valid
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


            #endregion

    my_callback = My_callback()
    #endregion

    #region train header layers
    if epoch_traintop > 0:
        op_adam_train_top = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        op_adam_train_top.compile(loss='mse',
                       optimizer=op_adam_train_top, metrics=['mae', 'mse'])
        from LIBS.CNN_Models.optimization.lookahead import Lookahead
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
            if dict_lr_traintop is not None:
                dict_lr = dict_lr_traintop

                for (k, v) in dict_lr.items():
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
            callbacks=[checkpointer_traintop, change_lr_traintop, my_callback])

    #endregion

    #region fine tuning all layers
    if epoch_finetuning > 0:
        model_finetuning = my_transfer_learning.convert_trainable_all(model_traintop)

        op_adam_finetuning = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model_finetuning.compile(loss='mse',
                                  optimizer=op_adam_finetuning, metrics=['mae', 'mse'])
        from LIBS.CNN_Models.optimization.lookahead import Lookahead
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
            try:
                file_object = open('lr.txt')
                line = file_object.readline()
                file_object.close()
                line = line.strip('\n') #删除换行符
                lr_rate = float(line)

                print("epoch：%d, set learning rate:  %f by lr.txt" % (epoch, lr_rate))
                K.set_value(model_finetuning.optimizer.lr, lr_rate)

            except Exception:
                if dict_lr_finetuning is not None:
                    dict_lr = dict_lr_finetuning

                    for (k, v) in dict_lr.items():
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
            callbacks=[checkpointer_finetuning, change_lr_finetuning, my_callback]
        )

    #endregion

    K.clear_session()  #release GPU memory


