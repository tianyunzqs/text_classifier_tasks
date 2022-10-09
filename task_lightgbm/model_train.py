# -*- coding: utf-8 -*-
# @Time        : 2022/10/8 18:43
# @Author      : tianyunzqs
# @Description :

import os
import json
import time
import jieba
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import metrics
from tqdm import tqdm
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            data.append((line_json['sentence'], int(line_json['label'])))
    return data


def train_model():
    # 导入数据
    train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
    dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
    # 导入停用词
    stopwords = load_stopwords(os.path.join(project_path, 'data', 'stopwords.txt'))

    train_x = [' '.join([word for word in jieba.lcut(d[0]) if word not in stopwords]) for d in train_data]
    train_y = [d[1] for d in train_data]

    print('开始训练模型...')
    t1 = time.time()
    # 将得到的词语转换为词频矩阵
    vectorizer = CountVectorizer(max_features=5000)
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 训练tfidf模型
    train_tfidf = transformer.fit_transform(vectorizer.fit_transform(np.array(train_x)))
    # 训练模型
    lgb_train = lgb.Dataset(train_tfidf.toarray(), train_y)

    # 构建lightGBM模型
    params = {
        'max_depth': 5,
        'num_leaves': 30,
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'objective': 'multiclass',
        'num_class': len(set(train_y)),
        'verbose': -1
    }
    # 设置迭代次数，默认为100，通常设置为100+
    num_boost_round = 200
    # 训练 lightGBM模型
    model = lgb.train(params, lgb_train, num_boost_round, verbose_eval=100)
    # 保存模型到文件
    # gbm.save_model('data/lightGBM_model')
    t2 = time.time()
    print('模型训练完成，耗时:{0}s'.format((t2 - t1)))

    # 保存模型
    model_path = './models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with open(os.path.join(model_path, 'model.pickle'), 'wb') as f:
        pickle.dump((vectorizer, transformer, model), f)

    # 验证集
    dev_x = [' '.join([word for word in jieba.lcut(d[0]) if word not in stopwords]) for d in dev_data]
    dev_y = [d[1] for d in dev_data]
    dev_tfidf = transformer.transform((vectorizer.transform(dev_x)))
    y_pred = model.predict(dev_tfidf, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)  # 获得最大概率对应的标签
    # y_pred = model.predict(dev_tfidf)
    # 输出各类别测试测试参数
    classify_report = metrics.classification_report(dev_y, y_pred, digits=4)
    with open(os.path.join(model_path, 'pred.txt'), 'w', encoding='utf-8') as f:
        f.write(classify_report)
        f.write('\n')
        f.write('训练耗时:{0}s'.format((t2 - t1)))
    print(classify_report)


if __name__ == '__main__':
    train_model()
