# coding=utf-8

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

# 数据加载过程
print('数据加载')
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
classes = 10
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
Y_train = to_categorical(y_train, classes)
Y_test = to_categorical(y_test, classes)

print('数据加载结束')
# 输入层
inputs = keras.Input(shape=[28, 28, 1])
# 卷积层1
conv1 = keras.layers.Conv2D(filters=64, kernel_size=[11, 11], strides=[1, 1], activation=keras.activations.relu,
                            use_bias=True, padding='same')(inputs)
# 标准化层1
stand1 = keras.layers.BatchNormalization(axis=1)(conv1)
#  池化层1
pooling1 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(stand1)

# 卷积层2
conv2 = keras.layers.Conv2D(filters=192, kernel_size=[5, 5], strides=[1, 1], activation=keras.activations.relu,
                            use_bias=True, padding='same')(pooling1)
# 标准化层2
stand2 = keras.layers.BatchNormalization(axis=1)(conv2)
#  池化层2
pooling2 = keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(stand2)

# 卷积层3
conv3 = keras.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu,
                            use_bias=True, padding='same')(pooling2)

# 卷积层4
conv4 = keras.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu,
                            use_bias=True, padding='same')(conv3)

# 卷积层5
conv5 = keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu,
                            use_bias=True, padding='same')(conv4)

# 池化层3
pooling3 = keras.layers.MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv5)

# 平缓层，将多维数据压成一维
flatten = keras.layers.Flatten()(pooling3)

# 全连接层1
fc1 = keras.layers.Dense(4096, activation=keras.activations.relu, use_bias=True)(flatten)
# 随机失活层1 防止过拟合
drop1 = keras.layers.Dropout(0.5)(fc1)

# 全连接层2
fc2 = keras.layers.Dense(4096, activation=keras.activations.relu, use_bias=True)(drop1)
# 随机失活层2
drop2 = keras.layers.Dropout(0.5)(fc2)

# 全连接层3
fc3 = keras.layers.Dense(10, activation=keras.activations.softmax, use_bias=True)(drop2)

model = keras.Model(inputs=inputs, outputs=fc3)
model.compile(optimizer=tf.optimizers.Adam(0.001), loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

# model.fit(X_train, Y_train, epochs=10, batch_size=64)
# model.save_weights('alex.h5')
model.load_weights('alex.h5')
score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

