import DataTools as dt
import ModelTools as mt
import AlexNetModel
from pathlib import Path


def check_file(path_name):
    model1_file = Path(path_name)
    if model1_file.exists():
        return True

    return False


def easy_model(origin_model_path, new_model_path):
    origin_model = AlexNetModel.origin_easy_model()

    if check_file(origin_model_path):
        origin_model = mt.load_model(origin_model_path)
    else:
        origin_model = mt.model_train(origin_model, first_train, first_label, origin_model_path, 10, 64)
    mt.model_evaluate(origin_model, test_img, test_label)

    # 第二次模型训练
    second_model = mt.model_update(origin_model_path, 11)
    second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 10, 64)
    mt.model_evaluate(second_model, test_img, test_label)


def complicated_model(origin_model_path, new_model_path):
    origin_model = AlexNetModel.origin_complicated_model()

    if check_file(origin_model_path):
        origin_model = mt.load_model(origin_model_path)
    else:
        origin_model = mt.model_train(origin_model, first_train, first_label, origin_model_path, 10, 64)
    mt.model_evaluate(origin_model, test_img, test_label)

    # 第二次模型训练
    second_model = mt.model_update(origin_model_path, 11)
    second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 10, 64)
    mt.model_evaluate(second_model, test_img, test_label)


def test_layer(origin_model_path, new_model_path):
    origin_model = AlexNetModel.origin_complicated_model()

    if check_file(origin_model_path):
        origin_model = mt.load_model(origin_model_path)
    else:
        origin_model = mt.model_train(origin_model, first_train, first_label, origin_model_path, 10, 64)
    mt.model_evaluate(origin_model, test_img, test_label)

    for layer_number in range(0, len(origin_model.layers), -1):
        second_model = mt.model_update(origin_model_path, layer_number)
        second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 10, 64)
        mt.model_evaluate(second_model, test_img, test_label)


if __name__ == '__main__':
    data_path = '../data/mnist'
    origin_model_path = '../weights/easyModel1.h5'
    new_model_path = '../weights/easyModel2.h5'
    origin_model_path_update = '../weights/easyModel3.h5'

    origin_complicated_path = '../weights/comModel1.h5'
    new_complicated_path = '../weights/comModel2.h5'
    origin_complicated_path_update = '../weights/comModel3.h5'

    first_train, first_label, second_train, second_label = dt.gene_train_data(7, data_path, 12000)
    test_img, test_label = dt.gene_test_data(data_path)
    print('first_train shape:', first_train.shape)
    print('first_label shape:', first_label.shape)
    print(first_train.shape[0], 'train samples')

    print('second_train shape:', second_train.shape)
    print('second_label shape:', second_label.shape)
    print(second_train.shape[0], 'train samples')

    easy = AlexNetModel.origin_easy_model()
    com = AlexNetModel.origin_complicated_model()

    easy.summary()

    com.summary()

    # # print('----------------easy_model run now-----------------')
    # # easy_model(origin_model_path, new_model_path)
    #
    # print('----------------complicated_model run now-----------------')
    # complicated_model(origin_complicated_path, new_complicated_path)
    # #
    # # print('----------------origin easy update run now--------------------')
    # # origin_model = mt.model_load(origin_model_path)
    # # origin_model = mt.model_train(origin_model, second_train, second_label, origin_model_path_update, 10, 64)
    # # mt.model_evaluate(origin_model, test_img, test_label)
    # #
    # # print('----------------origin complicated_model update run now--------------------')
    # # origin_model = mt.model_load(origin_complicated_path)
    # # origin_model = mt.model_train(origin_model, second_train, second_label, origin_complicated_path_update, 10, 64)
    # # mt.model_evaluate(origin_model, test_img, test_label)