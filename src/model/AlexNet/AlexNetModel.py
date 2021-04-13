# coding=utf-8
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Input, Dropout
from tensorflow import keras


def origin_easy_model():
    ## 模型的搭建
    model = Sequential([
        Input(shape=[28, 28, 1], name='inputs'),
        Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv1'),
        BatchNormalization(axis=1, name='bn1'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name="pool1"),
        Conv2D(filters=32, kernel_size=[4, 4], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv2'),
        BatchNormalization(axis=1, name='bn2'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool2'),
        Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv3'),
        Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv4'),
        Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv5'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool3'),
        Flatten(name='flatten'),
        Dense(256, activation=keras.activations.relu, use_bias=True, name='fc1'),
        Dropout(0.5, name='dropout1'),
        Dense(256, activation=keras.activations.relu, use_bias=True, name='fc2'),
        Dropout(0.5, name='dropout2'),
        Dense(10, activation=keras.activations.softmax, use_bias=True, name='fc3')
    ])

    ## 模型的编译
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def origin_complicated_model():
    ## 模型的搭建
    model = Sequential([
        Input(shape=[28, 28, 1], name='inputs'),
        Conv2D(filters=64, kernel_size=[11, 11], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv1'),
        BatchNormalization(axis=1, name='bn1'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name="pool1"),
        Conv2D(filters=192, kernel_size=[5, 5], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv2'),
        BatchNormalization(axis=1, name='bn2'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool2'),
        Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv3'),
        Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv4'),
        Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv5'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool3'),
        Flatten(name='flatten'),
        Dense(4096, activation=keras.activations.relu, use_bias=True, name='fc1'),
        Dropout(0.5, name='dropout1'),
        Dense(4096, activation=keras.activations.relu, use_bias=True, name='fc2'),
        Dropout(0.5, name='dropout2'),
        Dense(10, activation=keras.activations.softmax, use_bias=True, name='fc3')
    ])

    ## 模型的编译
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def demo_model():
    ## 模型的搭建
    model = Sequential([
        Input(shape=[28, 28, 1], name='inputs'),
        Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv1'),
        BatchNormalization(axis=1, name='bn1'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name="pool1"),
        Conv2D(filters=32, kernel_size=[4, 4], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv2'),
        BatchNormalization(axis=1, name='bn2'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool2'),
        Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same', name='conv3'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool3'),
        Flatten(name='flatten'),
        Dense(10, activation=keras.activations.relu, use_bias=True, name='fc1'),
        Dropout(0.5, name='dropout2'),
        Dense(10, activation=keras.activations.softmax, use_bias=True, name='fc3')
    ])

    ## 模型的编译
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
