from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, DepthwiseConv2D, ZeroPadding2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense, Reshape
import os
from keras import backend as K, Input, Model

num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 16

train_data_dir = './data/train2'
validation_data_dir = './data/validation'

# Let's use some data augmentaiton
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


def relu6(x):
    return keras.activations.relu(x, max_value=6)

    # def conv(input, kernel, strides):
    #     x = Conv2D(filters=32, kernel_size=kernel, strides=strides)(input)
    #     x = BatchNormalization()(x)
    #     x = Activation(activation=relu6)(x)
    #     return x
    #
    #
    # def bottleneck(x,filters, kernel):
    #     m = Conv2D(strides=(1, 1))(x)
    #     m = BatchNormalization()(m)
    #     m = Activation('relu6')(m)
    #     m = DepthwiseConv2D((3, 3))(m)
    #     m = BatchNormalization()(m)
    #     m = Activation('relu6')(m)
    #     m = Conv2D(strides=(1, 1))(m)
    #     m = BatchNormalization()(m)
    #     return Add()([m, x])
    #
    #
    # def MobileNet(input_shape):
    #     input = Input(input_shape)
    #     x = conv(input, (3, 3), (2, 2))
    #     x = bottleneck(x, 16, (3, 3))
    #     x = bottleneck(x, 24, (3, 3))
    #     x = bottleneck(x, 32, (3, 3))
    #     x = bottleneck(x, 64, (3, 3))
    #     x = bottleneck(x, 96, (3, 3))
    #     x = bottleneck(x, 160, (3, 3))
    #     x = bottleneck(x, 320, (3, 3))
    #     x = conv(x, (3, 3), (2, 2))
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dropout(0.5, name='Dropout')(x)
    #     x = Conv2D(6, (1, 1), padding='same')(x)
    #     x = Activation('softmax', name='softmax')(x)
    #     output = Reshape((6,))(x)
    #     model = Model(input, output)
    #     return model

def inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # Expand
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)
    if stride == 2:
        x = ZeroPadding2D()(x)

    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        activation=None,
                        padding='same' if stride == 1 else 'valid')(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None)(x)
    x = BatchNormalization()(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add()([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



def MobileNet_v2(input_shape, nb_class):
    alpha = 1.0
    inputs = Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D()(inputs)
    x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    x = Conv2D(1280, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=inputs, outputs=x)

    x = Reshape((1, 1, 1280))(model.output)
    x = Dropout(0.3)(x)
    x = Conv2D(nb_class, (1, 1), padding='same')(x)

    x = Activation('softmax')(x)
    output = Reshape((nb_class,))(x)
    model = Model(inputs=inputs, outputs=output)

    return model


# run function
model = MobileNet_v2(input_shape=(img_rows, img_cols,1), nb_class=6)


print(model.summary())
#
# checkpoint = ModelCheckpoint("emotion_MobileNet2.h5",
#                              monitor="val_loss",
#                              mode="min",
#                              save_best_only=True,
#                              verbose=1)
#
# earlystop = EarlyStopping(monitor='val_loss',
#                           min_delta=0,
#                           patience=4,
#                           verbose=1,
#                           restore_best_weights=True)
#
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
#
# # we put our call backs into a callback list
# callbacks = [earlystop, checkpoint, reduce_lr]
#
# # We use a very small learning rate
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=0.001),
#               metrics=['accuracy'])
#
# nb_train_samples = 29045
# nb_validation_samples = 3534
# epochs = 10
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
#
# model.save('MobileNet_save2.model')
