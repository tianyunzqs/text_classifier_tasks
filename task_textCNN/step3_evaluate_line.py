# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 在线预测

import os
import sys
import pickle
import json
import time

import tensorflow as tf
import numpy as np
import jieba.posseg as pseg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_cnn import TextCNN


def load_stopwords(path):
    stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
    return stopwords


stopwords = load_stopwords(r"../sample_data/stopwords.txt")
with open("./config_file", encoding="utf8") as f:
    config = json.load(f)


def load_model():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=config['allow_soft_placement'],
            log_device_placement=config['log_device_placement'])
        sess = tf.Session(config=session_conf)
        with open("./models/vocab33", 'rb') as f:
            vocab_processor = pickle.loads(f.read())
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=config['sequence_length'],
                num_classes=config['num_classes'],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=config['embedding_dim'],
                filter_sizes=list(map(int, config['filter_sizes'].split(","))),
                num_filters=config['num_filters'],
                l2_reg_lambda=config['l2_reg_lambda'])

            saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
    return vocab_processor, cnn, sess


vocab_processor, cnn, sess = load_model()


def evaluate_line(text):
    segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in stopwords])
    data = np.array(list(vocab_processor.transform([segwords])))
    feed_dict = {
        cnn.input_x: data,
        cnn.input_y: np.array([[0] * config['num_classes']]),
        cnn.dropout_keep_prob: 1.0
    }
    predictions = sess.run(cnn.predictions, feed_dict)
    label = config['tags'][str(predictions[0])]
    return label


if __name__ == '__main__':
    # 财经
    text = """交银货币清明假期前两日暂停申购和转换入全景网3月30日讯 交银施罗德基金周一公告称，公司旗下的交银施罗德货币市场证券投资基金将于2009年"清明"假期前两日暂停申购和转换入业务。公告表示，交银施罗德货币将于2009年4月2日、3日两天暂停办理基金的申购和转换入业务。转换出、赎回等其他业务以及公司管理的其他开放式基金的各项交易业务仍照常办理。自2009年4月7日起，所有销售网点恢复办理基金的正常申购和转换入业务。(全景网/雷鸣)"""
    t1 = time.time()
    label = evaluate_line(text=text)
    t2 = time.time()
    print(label)
    print('cost time: {0}ms'.format(t2 - t1))
