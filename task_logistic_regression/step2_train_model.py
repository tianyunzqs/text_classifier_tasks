# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 逻辑回归（Logistic Regression）——文本分类

import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics


def train_model(train_data, test_data):
    # 将得到的词语转换为词频矩阵
    vectorizer = TfidfVectorizer(min_df=1)
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    train_x, train_y = [], []
    for cls, data in train_data.items():
        train_x += data
        train_y += [cls] * len(data)
    # 训练tfidf模型
    train_tfidf = transformer.fit_transform(vectorizer.fit_transform(np.array(train_x)))
    # 训练逻辑回归模型
    lr_cls = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_tfidf, train_y)
    pickle.dump((vectorizer, transformer, lr_cls), open("./model.pickle", "wb"))

    # 预测测试数据
    y_true, y_pred = [], []
    for cls, data in test_data.items():
        for d in data:
            test_tfidf = transformer.transform(vectorizer.transform([d]))
            prediction = lr_cls.predict(test_tfidf)[0]
            y_pred.append(prediction)
            y_true.append(cls)
    # 输出各类别测试测试参数
    classify_report = metrics.classification_report(y_true, y_pred)
    print(classify_report)


if __name__ == '__main__':
    train = pickle.load(open("./train_data.pickle", "rb"))
    test = pickle.load(open("./test_data.pickle", "rb"))
    train_model(train_data=train, test_data=test)
