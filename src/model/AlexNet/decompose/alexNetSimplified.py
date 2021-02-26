# coding=utf-8

import numpy as np

np.random.seed(1337)  # 为了再见性
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Input, Dropout
from keras.utils import to_categorical
from keras import backend as K
from tensorflow import keras

img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

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

# 转换为one_hot类型
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

## 模型的搭建

model = Sequential([
    Input(shape=[28, 28, 1], name='inputs'),
    Conv2D(filters=8, kernel_size=[5, 5], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
           padding='same', name='conv1'),
    BatchNormalization(axis=1, name='bn1'),
    MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name="pool1"),
    Conv2D(filters=16, kernel_size=[4, 4], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
           padding='same', name='conv2'),
    BatchNormalization(axis=1, name='bn2'),
    MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool2'),
    Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
           padding='same', name='conv3'),
    Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
           padding='same', name='conv4'),
    Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
           padding='same', name='conv5'),
    MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid', name='pool3'),
    Flatten(name='flatten'),
    Dense(128, activation=keras.activations.relu, use_bias=True, name='fc1'),
    Dropout(0.5, name='dropout1'),
    Dense(128, activation=keras.activations.relu, use_bias=True, name='fc2'),
    Dropout(0.5, name='dropout2'),
    Dense(10, activation=keras.activations.softmax, use_bias=True, name='fc3')
])

## 模型的编译
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

## 模型的第一次训练
model.fit(X_train[:30000],Y_train[:30000],epochs=5,batch_size=64)
model.save('../weights/model1.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

del model

## 加载训练好的模型，准备进行第二次训练
model = load_model('../weights/model1.h5')
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# 查看并设置每层是否可训练
print("-------------------------------------------------")
for layer in model.layers[:12]:
    print(layer.name, ' is trainable? ', layer.trainable)
    layer.trainable = False

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 检验并查看是否设置成功
print("-------------------------------------------------")
for layer in model.layers:
    print(layer.name, ' is trainable? ', layer.trainable)

## 模型的第二次训练
model.fit(X_train[30000:], Y_train[30000:], epochs=5, batch_size=64)
model.save('../weights/model2.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
predictions = model.predict(X_test[:5])
print(np.argmax(predictions, axis=1))
print(np.argmax(Y_test[:5], axis=1))
'''
