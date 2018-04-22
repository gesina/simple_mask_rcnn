#! /bin/python3

from __future__ import print_function
import keras
import mnist_mask
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K

# CONFIG
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 4
IMG_ROWS, IMG_COLS = 28, 28
if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
else:
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

(x_train, y_train, mask_train), (x_test, y_test, mask_test) = mnist_mask.load_data()

# INPUT
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
    x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
else:
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


# MODEL
model = Sequential([
    Conv2D(
        input_shape=INPUT_SHAPE,
        filters=32,
        kernel_size=(4, 4),  # should be divisible by stride
        strides=(2, 2),
        padding="valid", # same takes ridiculously much longer
        activation="relu",
    ),
    #MaxPooling2D(pool_size=(2,2)),
    Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="valid",
        activation="relu",
    ),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(rate=0.25),
    Flatten(),
    Dense(units=64, activation="relu"),
    Dropout(rate=0.5),
    Dense(units=NUM_CLASSES, activation="softmax")
])
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

# TRAIN
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))

# EVAL
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])