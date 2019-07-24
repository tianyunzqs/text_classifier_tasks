# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 在线预测

import time
import pickle

import jieba.posseg as pseg


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


vectorizer, transformer, dt_model = pickle.load(open("./model.pickle", "rb"))
stopwords = load_stopwords(r"../sample_data/stopwords.txt")


def evaluate_line(text):
    segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in stopwords])
    tfidf_vec = transformer.transform(vectorizer.transform([segwords]))
    prediction_label = dt_model.predict(tfidf_vec)[0]
    return prediction_label


if __name__ == '__main__':
    # 财经
    text = """交银货币清明假期前两日暂停申购和转换入全景网3月30日讯 交银施罗德基金周一公告称，公司旗下的交银施罗德货币市场证券投资基金将于2009年"清明"假期前两日暂停申购和转换入业务。公告表示，交银施罗德货币将于2009年4月2日、3日两天暂停办理基金的申购和转换入业务。转换出、赎回等其他业务以及公司管理的其他开放式基金的各项交易业务仍照常办理。自2009年4月7日起，所有销售网点恢复办理基金的正常申购和转换入业务。(全景网/雷鸣)"""
    t1 = time.time()
    label = evaluate_line(text=text)
    t2 = time.time()
    print(label)
    print('cost time: {0}ms'.format(t2 - t1))
