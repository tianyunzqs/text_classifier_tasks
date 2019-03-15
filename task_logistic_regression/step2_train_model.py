# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 

import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


def train_model(train_data, test_data):
    # 将得到的词语转换为词频矩阵
    vectorizer = TfidfVectorizer(min_df=1)
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    train_x, train_y = [], []
    for cls, data in train_data.items():
        train_x += data
        train_y += [cls] * len(data)

    tfidf = transformer.fit_transform(vectorizer.fit_transform(np.array(train_x)))

    # nb_model = BernoulliNB()
    # nb_model.fit(tfidf, train_y)
    nb_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(tfidf, train_y)
    # pickle.dump((vectorizer, transformer, nb_model), open("./model.pickle", "wb"))

    for cls, data in test_data.items():
        for d in data:
            tfidf2 = transformer.transform(vectorizer.transform([d]))
            pred2 = nb_model.predict(tfidf2)
            print(pred2)
            break
        break


if __name__ == '__main__':
    train = pickle.load(open("./train_data.pickle", "rb"))
    test = pickle.load(open("./test_data.pickle", "rb"))
    train_model(train_data=train, test_data=test)
