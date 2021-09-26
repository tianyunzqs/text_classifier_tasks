# -*- coding: utf-8 -*-
# @Time        : 2021/9/26 14:26
# @Author      : tianyunzqs
# @Description :

import os
import time
import json
import fasttext
import jieba
from tqdm import tqdm
from sklearn import metrics
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


def prepare_fasttext_data(path, mode='train'):
    labels = set()
    with open(path, "r", encoding="utf-8") as f, open('{0}.txt'.format(mode), 'w', encoding='utf-8') as f1:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            label = '__label__' + line_json['label']
            labels.add(line_json['label'])
            content = ' '.join(jieba.lcut(line_json['sentence']))
            f1.write(content + '\t' + label + '\n')
    return labels


# labels = prepare_fasttext_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'), 'train')
# num_classes = len(labels)
# dev_data = prepare_fasttext_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'), 'dev')

model_path = './models'
if not os.path.exists(model_path):
    os.mkdir(model_path)
model = fasttext.train_supervised('train.txt',
                                  wordNgrams=2,
                                  lr=1e-1,
                                  epoch=300,
                                  dim=200,
                                  label_prefix="__label__")
model.save_model(os.path.join(model_path, "model_filename.ftz"))
result = model.test('dev.txt')
print(result)
# print("P@1:", result.precision)  # 准确率
# print("R@2:", result.recall)     # 召回率


# def train_model(train_data_path, test_data_path, model_save_path):
#     t1 = time.time()
#     classifier = fasttext.supervised(train_data_path, model_save_path, label_prefix="__label__")
#     t2 = time.time()
#     print('train model over. it took {0}ms'.format((t2 - t1)))
#
#     result = classifier.test(test_data_path)
#     print("P@1:", result.precision)  # 准确率
#     print("R@2:", result.recall)     # 召回率
#     print("Number of examples:", result.nexamples)  # 预测的样本数量
#
#     # 预测测试数据
#     y_true, y_pred = [], []
#     with open(test_data_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             parts = line.split(' , ')
#             if len(parts) != 2:
#                 continue
#             cls, txt = parts[0], parts[1]
#             prediction = classifier.predict([txt])
#             y_pred.append(prediction[0][0])
#             y_true.append(cls.replace('__label__', '').strip())
#
#     # 输出各类别测试测试参数
#     print(y_true[:10], y_pred[:10])
#     classify_report = metrics.classification_report(y_true, y_pred)
#     print(classify_report)
#
#
# if __name__ == '__main__':
#     train_model(train_data_path='./train_data.txt', test_data_path='./test_data.txt', model_save_path='./fasttext.model')
