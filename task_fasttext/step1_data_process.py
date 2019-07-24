# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:09
# @Author      : tianyunzqs
# @Description : 数据预处理

import json
import random

import jieba.posseg as pseg
from tqdm import tqdm


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


def load_data(path, stopwords):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            classify, text = parts[0], parts[1]
            classify = '__label__' + classify
            segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in stopwords])
            data.append(classify + ' , ' + segwords)
            # 用列表形式，可以打乱顺序，不至于使得训练样本类别集中
            # if classify not in data:
            #     data[classify] = [segwords]
            # else:
            #     data[classify].append(segwords)
    return data


# def load_data(path, stopwords):
#     data = []
#     with open(path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in tqdm(lines):
#             line_json = json.loads(line)
#             classify = '__label__' + line_json["classify"]
#             text = line_json["title"] + " " + line_json["content"]
#             segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in stopwords])
#             data.append(classify + ' , ' + segwords)
#             # 用列表形式，可以打乱顺序，不至于使得训练样本类别集中
#             # if classify not in data:
#             #     data[classify] = [segwords]
#             # else:
#             #     data[classify].append(segwords)
#     return data


def write_fasttext_format_file(data, path):
    random.shuffle(data)
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(d)
            f.write('\n')


if __name__ == '__main__':
    stopwords = load_stopwords(r"../sample_data/stopwords.txt")
    train_path = r"../sample_data/cnews.train.txt"
    train_data = load_data(train_path, stopwords=stopwords)
    test_path = r"../sample_data/cnews.test.txt"
    test_data = load_data(test_path, stopwords=stopwords)
    write_fasttext_format_file(data=train_data, path='./train_data.txt')
    write_fasttext_format_file(data=test_data, path='./test_data.txt')
