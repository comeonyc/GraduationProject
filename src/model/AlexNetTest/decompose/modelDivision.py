import numpy as np
import os
import gzip

from keras.models import Sequential, load_model,Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Input, Dropout, InputLayer
from keras.utils import to_categorical
from keras import backend as K
from tensorflow import keras


def model_train(X_train, Y_train, X_test, Y_test):
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
    ## 模型的第一次训练
    model.fit(X_train[:5000], Y_train[:5000], epochs=5, batch_size=64)
    model.save('../weights/model_o.h5')
    #model = load_model('../weights/model_o.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model


def model_divide(model, level_number):
    model_first = Sequential()
    model_second = Sequential()

    model_first.add(InputLayer(input_shape=model.layers[0].input_shape[1:]))
    for current_layer in range(0, level_number + 1):
        model_first.add(model.layers[current_layer])

    model_second.add(InputLayer(input_shape=model.layers[level_number + 1].input_shape[1:]))
    for current_layer in range(level_number + 1, len(model.layers)):
        model_second.add(model.layers[current_layer])

    return model_first, model_second


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def prepared_data():
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    (X_train, y_train) = load_mnist(path='../../data/fashion', kind='train')
    (X_test, y_test) = load_mnist(path='../../data/fashion', kind='t10k')

    # 根据不同的backend定下不同的格式
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_train[0])

    # 转换为one_hot类型
    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test


def gene_data():
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    (X_train, y_train) = load_mnist(path='../../data/fashion', kind='train')
    (X_test, y_test) = load_mnist(path='../../data/fashion', kind='t10k')
    print(X_train)
    data_1_train = []
    data_1_label = []
    data_2_train = []
    data_2_label = []
    for cur_y in range(0, len(y_train)):
        if 0 <= y_train[cur_y] <= 7:
            data_1_train.append(X_train[cur_y])
            data_1_label.append(y_train[cur_y])
        else:
            data_2_train.append(X_train[cur_y])
            data_2_label.append(y_train[cur_y])
    data_1_train = np.array(data_1_train)
    data_1_label = np.array(data_1_label)
    data_2_train = np.array(data_2_train)
    data_2_label = np.array(data_2_label)

    # 根据不同的backend定下不同的格式
    if K.image_data_format() == 'channels_first':
        data_1_train = data_1_train.reshape(data_1_train.shape[0], 1, img_rows, img_cols)
        data_2_train = data_2_train.reshape(data_2_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        data_1_train = data_1_train.reshape(data_1_train.shape[0], img_rows, img_cols, 1)
        data_2_train = data_2_train.reshape(data_2_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    data_1_train = data_1_train.astype('float32')
    data_2_train = data_2_train.astype('float32')
    X_test = X_test.astype('float32')
    data_1_train /= 255
    data_2_train /= 255
    X_test /= 255

    data_1_label = to_categorical(data_1_label,10)
    data_2_label = to_categorical(data_2_label,10)
    # # 转换为one_hot类型
    # Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return data_1_train, data_1_label, data_2_train, data_2_label, X_test, Y_test


if __name__ == '__main__':
    data_1_train, data_1_label, data_2_train, data_2_label, X_test, Y_test = gene_data()


    # X_train, Y_train, X_test, Y_test = prepared_data()

    # model = model_train(X_train, Y_train, X_test, Y_test)
    #
    # model_first, model_second = model_divide(model, 10)
    #
    # print('获取训练集 上层输出结果')
    # first_train_output = model_first.predict(X_train[5000:60000])
    #
    # model.summary()
    # model_first.summary()
    # model_second.summary()
    #
    # model_second.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model_second.save('../weights/model_s_1.h5')
    # model_second.fit(first_train_output, Y_train[5000:60000], epochs=5, batch_size=64)
    # model_second.save('../weights/model_s.h5')
    #
    # print('获取测试集 上层输出结果')
    # first_test_output = model_first.predict(X_test)
    # score = model_second.evaluate(first_test_output, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
