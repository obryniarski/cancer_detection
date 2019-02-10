from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

import keras
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def CNN(input_shape=(96, 96, 3)):

    first_filters = 64
    second_filters = 128
    third_filters = 256
    fourth_filters = 256
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()

    model.add(Conv2D(first_filters, kernel_size, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(first_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    # model.add(Dropout(0.2))

    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    # model.add(Dropout(0.2))

    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size))

    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # # model.add(MaxPooling2D(pool_size))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(1000, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=3e-4),
                    metrics=['accuracy'])

    return model


def CNN_experiment(value, input_shape=(96, 96, 3)):
    first_filters = 32
    second_filters = 64
    third_filters = 128
    fourth_filters = 256
    kernel_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()

    model.add(Conv2D(first_filters, kernel_size, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(first_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.2))

    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(second_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.2))

    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(third_filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.2))

    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size))

    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(fourth_filters, kernel_size, activation='relu'))
    # model.add(BatchNormalization())
    # # model.add(MaxPooling2D(pool_size))

    model.add(Flatten())
    model.add(Dense(4096 * (2 ** value), activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(4096 * (2 ** value), activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(800 * (2 ** value), activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=3e-4),
                  metrics=['accuracy'])

    return model