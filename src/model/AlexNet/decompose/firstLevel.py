from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Input, Dropout
from keras.utils import to_categorical
from tensorflow import keras


def model_train(inputs):
    model = Sequential([
        Input(shape=[28, 28, 1]),
        Conv2D(filters=8, kernel_size=[5, 5], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid'),
        Conv2D(filters=16, kernel_size=[4, 4], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid'),
        Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same'),
        Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same'),
        Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], activation=keras.activations.relu, use_bias=True,
               padding='same'),
        MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid'),
        Flatten(),
        Dense(128, activation=keras.activations.relu, use_bias=True),
        Dropout(0.5),
        Dense(128, activation=keras.activations.relu, use_bias=True),
        Dropout(0.5),
        Dense(10, activation=keras.activations.softmax, use_bias=True)
    ])

    ## 模型的编译
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    ## 模型的第一次训练
    (trainImage, trainLabel), (testImage, testLabel) = inputs[:30000]
    model.fit(trainImage, trainLabel, epochs=5, batch_size=64)
    model.save('../weights/model1.h5')
    score = model.evaluate(testImage, testLabel, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def get_second_inputs(model,inputs):
    ## 得到模型和数据
    model = load_model('../weights/model1.h5')
    (trainImage, trainLabel), (testImage, testLabel) = inputs[30000:]

    #for (trainImage, trainLabel) in len(trainImage)
