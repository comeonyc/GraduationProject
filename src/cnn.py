# coding=utf-8

import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from keras.utils import to_categorical

# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

## 规范化数据，将图像像素值从[0,255] 规范到 [-0.5,0.5]
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

## 将数据规范化三维数据，keras处理图像是3维的数据
train_images = np.expand_dims(train_images,axis=3)
test_images = np.expand_dims(test_images,axis=3)

## 模型的搭建
num_filters = 8
filter_size = 3
pool_size = 2
model = Sequential([
    Conv2D(num_filters,filter_size,input_shape=(28,28,1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10,activation='softmax')
])

# ## 模型的编译
# model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
#
# ## 模型的训练
# model.fit(train_images,to_categorical(train_labels),epochs=5,batch_size=32)
#
# model.save_weights('model.h5')

model.load_weights('model.h5')
predictions = model.predict(test_images[:5])
print(np.argmax(predictions,axis=1))
print(test_labels[:5])

print('mac分支修改')
