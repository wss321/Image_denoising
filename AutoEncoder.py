from keras import Model
import keras
from keras.layers import Input, Convolution2D, MaxPooling2D, BatchNormalization, Activation, \
    UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping
from keras import regularizers
from keras.models import load_model
from dataset import load_train_data_1C as load_train
from dataset import load_test_data_1C as load_test
from dataset2 import load_train_data_1C as load_train_moisy
from dataset2 import load_test_data_1C as load_test_moisy
from dataset import visualize_image

import matplotlib.pyplot as plt


def _conv_act_bn(inputs, filters, kernel_size=(3, 3), norm_rate=0.0, padding='same'):
    x = Convolution2D(filters, kernel_size=kernel_size, padding=padding,
                      kernel_regularizer=regularizers.l2(norm_rate))(inputs)

    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    return x


def AutoEncoder():
    input_img = Input(shape=(256, 256, 1))
    x = _conv_act_bn(input_img, 64)
    x = _conv_act_bn(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = _conv_act_bn(x, 128)
    x = _conv_act_bn(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = _conv_act_bn(x, 256)
    x = _conv_act_bn(x, 256)
    x = _conv_act_bn(x, 256)
    x = MaxPooling2D((2, 2))(x)
    x = _conv_act_bn(x, 512)
    x = _conv_act_bn(x, 512)
    x = _conv_act_bn(x, 512)
    encoded = MaxPooling2D((2, 2))(x)

    x = _conv_act_bn(encoded, 512)
    x = _conv_act_bn(x, 512)
    x = _conv_act_bn(x, 512)
    x = UpSampling2D((2, 2))(x)
    x = _conv_act_bn(x, 256)
    x = _conv_act_bn(x, 256)
    x = _conv_act_bn(x, 256)
    x = UpSampling2D((2, 2))(x)
    x = _conv_act_bn(x, 128)
    x = _conv_act_bn(x, 128)
    x = UpSampling2D((2, 2))(x)
    x = _conv_act_bn(x, 64)
    x = _conv_act_bn(x, 64)
    x = UpSampling2D((2, 2))(x)
    decoded = x
    autoencoder = Model(input_img, decoded)
    return autoencoder


if __name__ == '__main__':
    tensorboard = TensorBoard(log_dir='./')
    losscalback = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', mode='auto',
                                 save_best_only='True')
    callback_lists = [earlystop, checkpoint, losscalback, tensorboard]
    autoencoder = AutoEncoder()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # training
    train_data, _ = load_train(shuffle_buffer_size=None)
    train_data_noisy, _ = load_train_moisy(shuffle_buffer_size=None)
    test_data, _ = load_test(shuffle_buffer_size=None)
    test_data_noisy, _ = load_test_moisy(shuffle_buffer_size=None)

    visualize_image(train_data[0].reshape(256, 256))
    visualize_image(train_data_noisy[0].reshape(256, 256))
    print((train_data_noisy[0] - train_data[0]).reshape(256, 256))

    autoencoder.fit(train_data_noisy, train_data,
                    epochs=10000,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(test_data_noisy, test_data),
                    callbacks=callback_lists)
    autoencoder = load_model('best_model.h5')
    decoded_imgs = autoencoder.predict(test_data_noisy)

    n = 20
    for i in range(n):
        # display noisy
        plt.figure(dpi=200)

        ax = plt.subplot(1, 3, 1)
        plt.title('{}-noisy'.format(i))
        plt.imshow(test_data_noisy[i].reshape(256, 256))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(1, 3, 2)
        plt.title('{}-reconstruction'.format(i))
        plt.imshow(decoded_imgs[i].reshape(256, 256))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display original
        ax = plt.subplot(1, 3, 3)
        plt.title('{}-original'.format(i))
        plt.imshow(test_data[i].reshape(256, 256))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
