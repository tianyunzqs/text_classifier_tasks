# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:09
# @Author      : tianyunzqs
# @Description : 数据预处理，原始数据来源：https://blog.csdn.net/m0_38088359/article/details/83004972

import json
import pickle

import jieba.posseg as pseg
from tqdm import tqdm


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


def load_data(path, stopwords):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            classify, text = parts[0], parts[1]
            segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in stopwords])
            if classify not in data:
                data[classify] = [segwords]
            else:
                data[classify].append(segwords)
    return data


# def load_data(path, stopwords):
#     data = {}
#     with open(path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in tqdm(lines):
#             line_json = json.loads(line)
#             classify = line_json["classify"]
#             text = line_json["title"] + " " + line_json["content"]
#             segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in stopwords])
#             if classify not in data:
#                 data[classify] = [segwords]
#             else:
#                 data[classify].append(segwords)
#     return data


if __name__ == '__main__':
    stopwords = load_stopwords(r"../sample_data/stopwords.txt")
    train_path = r"../sample_data/cnews.train.txt"
    train_data = load_data(train_path, stopwords=stopwords)
    test_path = r"../sample_data/cnews.test.txt"
    test_data = load_data(test_path, stopwords=stopwords)
    pickle.dump(train_data, open("./data/train_data.pickle", "wb"))
    pickle.dump(test_data, open("./data/test_data.pickle", "wb"))
