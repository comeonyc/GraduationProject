import DataTools as dt
import ModelTools as mt
import AlexNetModel
from pathlib import Path
import numpy as np
import pandas as pd


def check_file(path_name):
    model1_file = Path(path_name)
    if model1_file.exists():
        return True

    return False


if __name__ == '__main__':
    data_path = '../data/fashion'
    origin_model_path = '../weights/demo1.h5'
    new_model_path = '../weights/demo2.h5'

    # first_train, first_label, second_train, second_label = dt.gene_train_data(7, data_path, 12000)
    # test_img, test_label = dt.gene_test_data(data_path)
    # print('first_train shape:', first_train.shape)
    # print('first_label shape:', first_label.shape)
    # print(first_train.shape[0], 'train samples')
    #
    # print('second_train shape:', second_train.shape)
    # print('second_label shape:', second_label.shape)
    # print(second_train.shape[0], 'train samples')

    # origin_model = AlexNetModel.origin_easy_model()
    #
    # # if check_file(origin_model_path):
    # #     origin_model = mt.load_model(origin_model_path)
    # # else:
    # #     origin_model = mt.model_train(origin_model, first_train, first_label, origin_model_path, 5, 128)
    # # mt.model_evaluate(origin_model, test_img, test_label)
    # #
    # # # 第二次模型训练
    # # second_model = mt.model_update(origin_model_path, 11)
    # # second_model = mt.model_train(second_model, second_train, second_label, new_model_path, 5, 128)
    # # mt.model_evaluate(second_model, test_img, test_label)

    # origin_model = AlexNetModel.demo_model()
    # origin_model = mt.model_train(origin_model, first_train, first_label, origin_model_path, 2, 256)

    origin_model = mt.model_load(origin_model_path)

    first_model, second_model = mt.model_divide(model=origin_model, level_number=10)

    second_model.save(new_model_path)
    weight_Dense_1, bias_Dense_1 = origin_model.get_layer('fc3').get_weights()
    weight_Dense_2, bias_Dense_2 = second_model.get_layer('fc3').get_weights()
    print('--------原始模型：截断层与下层参数--------')
    print(weight_Dense_1)
    print('--------下层模型：输入层与第一层参数--------')
    print(weight_Dense_2)
    print('对比结果如下: ', (weight_Dense_1 == weight_Dense_2).all())

    data = pd.DataFrame(weight_Dense_1)
    writer = pd.ExcelWriter('A.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()

    data = pd.DataFrame(weight_Dense_2)
    data.to_excel(writer, 'page_2', float_format='%.5f')
    writer.save()
    writer.close()
