# coding=utf-8

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
np.random.seed(1337)  # 为了再见性


def loadData():
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (trainImage, trainLabel), (testImage, testLabel) = mnist.load_data()

    # 根据不同的backend定下不同的格式
    if K.image_data_format() == 'channels_first':
        trainImage = trainImage.reshape(trainImage.shape[0], 1, img_rows, img_cols)
        testImage = testImage.reshape(testImage.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainImage = trainImage.reshape(trainImage.shape[0], img_rows, img_cols, 1)
        testImage = testImage.reshape(testImage.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainImage = trainImage.astype('float32')
    testImage = testImage.astype('float32')
    trainImage /= 255
    testImage /= 255
    print('trainImage shape:', trainImage.shape)
    print(trainImage.shape[0], 'train samples')
    print(testImage.shape[0], 'test samples')

    # 转换为one_hot类型
    trainLabel = to_categorical(trainLabel, 10)
    testLabel = to_categorical(testLabel, 10)

    for (img,label) in (trainImage,trainLabel):
        print(img.shape)
        print(label)

    return (trainImage, trainLabel), (testImage, testLabel)

loadData()