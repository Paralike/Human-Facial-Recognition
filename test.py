from __future__ import print_function
import keras
import os
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
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

CATEGORIES = ["Angry", "Fear","Happy","Neutral", "Sad", "Suprise"]

num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 32

train_data_dir = './data/train2'
validation_data_dir = './data/validation'

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


def prepare(filepath):
    IMG_SIZE = 48  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = keras.models.load_model("first_save.model")

# img = cv2.imread('../data/validation/Angry/0.jpg',0) # reads image 'opencv-logo.png' as grayscale
# plt.imshow(img, cmap='gray')
# img_array = cv2.imread('data/validation/Angry/0.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("im",img_array)
# prediction = model.predict([prepare('data/validation/Happy/20.jpg')])
# print(prediction[0])  # will be a list in a list.
# print (CATEGORIES[np.where(prediction[0] == np.amax(prediction[0]))[0][0]])
# print(CATEGORIES[int(prediction.values(max(prediction[0][0])))])
# print(CATEGORIES[int(prediction[0][0])])

# files = {}
# # r=root, d=directories, f = files
# for r, d, f in os.walk(validation_data_dir):
#     # print(r)
#     for file in f:
#         if '.jpg' in file:
#             # print("jpg found")
#             if 'Angry' in file:
#                 files[os.path.join(r, file)] = 'Angry'
#             elif 'Fear' in file:
#                 files[os.path.join(r, file)] = 'Fear'
#             elif 'Happy' in file:
#                 files[os.path.join(r, file)] = 'Happy'
#             elif 'Neutral' in file:
#                 files[os.path.join(r, file)] = 'Neutral'
#             elif 'Sad' in file :
#                 files[os.path.join(r, file)]= 'Sad'
#             elif 'Suprise' in file:
#                 files[os.path.join(r, file)] = 'Suprise'
#
#             # files.append(os.path.join(r, file))
# print (files)
# for f in files:
#     print(f)

# probabilities = model.predict_generator(validation_generator, 10)
# print(probabilities)
# print(len(probabilities))

loss, acc = model.evaluate_generator(validation_generator, verbose=0)
print("loss: "+ str(loss) + " acc " + str(acc))