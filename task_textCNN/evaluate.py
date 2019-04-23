# -*- coding: utf-8 -*-
# @Time        : 2019/4/15 14:42
# @Author      : tianyunzqs
# @Description : 

import os
import sys
import pickle
import json

import tensorflow as tf
import numpy as np
from sklearn import metrics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_cnn import TextCNN


with open("./data/train_data.pickle", 'rb') as f:
    train_data = pickle.loads(f.read())
with open("./data/test_data.pickle", 'rb') as f:
    test_data = pickle.loads(f.read())

classify = {}
tags = {}
for i, cls in enumerate(train_data.keys()):
    tmp = [0] * len(train_data.keys())
    tmp[i] = 1
    classify[cls] = tmp
    tags[i] = cls
x_test, y_test = [], []
for cls, data in test_data.items():
    # data = data[:int(0.5 * len(data))]
    x_test.extend(data)
    y_test.extend([classify[cls]] * len(data))


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


def test_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }
    predictions, accuracy = sess.run([cnn.predictions, cnn.accuracy], feed_dict)
    return predictions, accuracy


y_true, y_pred, y_acc = [], [], 0
for x, y in zip(x_test, y_test):
    data = np.array(list(vocab_processor.transform([x])))
    pred, acc = test_step(data, np.array([y]))
    y_acc += acc
    y_true.append(tags[y.index(max(y))])
    y_pred.append(tags[pred[0]])
classify_report = metrics.classification_report(y_true, y_pred)
y_acc = y_acc / len(y_true)
print(classify_report)
print(y_acc)
