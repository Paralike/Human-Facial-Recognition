from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
import os

num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 1

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

model = Sequential()
# 1 LAYER
#CONV => RELU => CONV => RELU => POOL
model.add(Conv2D(64,(3,3),padding="same", input_shape=(img_rows,img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding="same", input_shape=(img_rows,img_cols,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#2 LAYER
#CONV => RELU => CONV => RELU => POOL
model.add(Conv2D(256,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#3LAYER
#CONV => RELU => CONV => RELU => CONV =>RELU => POOL
model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#4LAYER
#CONV => RELU => CONV => RELU => CONV =>RELU => POOL
model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#5LAYER
#CONV => RELU => CONV => RELU => CONV =>RELU => POOL
model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(512,(3,3),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#6 LAYER
#FULLY CONNECTED => RELU
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#7 LAYER
#FULLY CONNECTED => RELU
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#8 LAYER
#FULLY CONNECTED => RELU
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block #9: softmax classifier
model.add(Dense(num_classes))
model.add(Activation("softmax"))

print(model.summary())

# checkpoint = ModelCheckpoint("emotion_VGG_16.h5",
#                              monitor="val_loss",
#                              mode="min",
#                              save_best_only = True,
#                              verbose=1)
#
# earlystop = EarlyStopping(monitor = 'val_loss',
#                           min_delta = 0,
#                           patience = 4,
#                           verbose = 1,
#                           restore_best_weights = True)
#
# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)
#
# # we put our call backs into a callback list
# callbacks = [earlystop, checkpoint, reduce_lr]
#
# # We use a very small learning rate
# model.compile(loss = 'categorical_crossentropy',
#               optimizer = Adam(lr=0.001),
#               metrics = ['accuracy'])
#
# nb_train_samples = 29045
# nb_validation_samples = 3534
# epochs = 10
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch = nb_train_samples // batch_size,
#     epochs = epochs,
#     callbacks = callbacks,
#     validation_data = validation_generator,
#     validation_steps = nb_validation_samples // batch_size)
#
# model.save('VGG16_save.model')

