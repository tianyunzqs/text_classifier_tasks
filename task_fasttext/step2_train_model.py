# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 利用step1处理好的fasttext格式，训练fasttext分类模型

import time

import fasttext
from sklearn import metrics


def train_model(train_data_path, test_data_path, model_save_path):
    t1 = time.time()
    classifier = fasttext.supervised(train_data_path, model_save_path, label_prefix="__label__")
    t2 = time.time()
    print('train model over. it took {0}ms'.format((t2 - t1)))

    result = classifier.test(test_data_path)
    print("P@1:", result.precision)  # 准确率
    print("R@2:", result.recall)     # 召回率
    print("Number of examples:", result.nexamples)  # 预测的样本数量

    # 预测测试数据
    y_true, y_pred = [], []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(' , ')
            if len(parts) != 2:
                continue
            cls, txt = parts[0], parts[1]
            prediction = classifier.predict([txt])
            y_pred.append(prediction[0][0])
            y_true.append(cls.replace('__label__', '').strip())

    # 输出各类别测试测试参数
    print(y_true[:10], y_pred[:10])
    classify_report = metrics.classification_report(y_true, y_pred)
    print(classify_report)


if __name__ == '__main__':
    train_model(train_data_path='./train_data.txt', test_data_path='./test_data.txt', model_save_path='./fasttext.model')
