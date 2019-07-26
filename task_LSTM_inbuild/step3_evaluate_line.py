# -*- coding: utf-8 -*-
# @Time        : 2019/7/26 10:18
# @Author      : tianyunzqs
# @Description : 

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_LSTM_inbuild.data_helper import Vocab, CategoryDict
from task_LSTM_inbuild.text_lstm import LSTM_Model


# lstm 需要的参数
def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size = 16, # 每个词语的向量的长度

        # 指定 lstm 的 步长， 一个sentence中会有多少个词语
        # 因为执行的过程中是用的minibatch，每个batch之间还是需要对齐的
        # 在测试时，可以是一个变长的
        num_timesteps = 50,  # 在一个sentence中 有 50 个词语

        num_lstm_nodes = [32, 32], # 每一层的size是多少
        num_lstm_layers = 2, # 和上句的len 是一致的
        # 有 两层 神经单元，每一层都是 32 个 神经单元

        num_fc_nodes = 32, # 全连接的节点数
        batch_size = 100,
        clip_lstm_grads = 1.0,
        # 控制lstm的梯度，因为lstm很容易梯度爆炸或者消失
        # 这种方式就相当于给lstm设置一个上限，如果超过了这个上限，就设置为这个值
        learning_rate = 0.001,
        num_word_threshold = 10, # 词频太少的词，对于模型训练是没有帮助的，因此设置一个门限
    )


hps = get_default_params()  # 生成参数对象


def load_model():
    vocab_file = 'D:/alg_file/data/cnews/cnews.vocab.txt'
    category_file = 'D:/alg_file/data/cnews/cnews.category.txt'
    vocab = Vocab(vocab_file, hps.num_word_threshold)
    category = CategoryDict(category_file)

    graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
    sess = tf.Session(graph=graph)  # 创建新的sess
    with sess.as_default():
        with graph.as_default():
            lstm = LSTM_Model(hps, vocab.size(), category.size())

            saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

    return vocab, category, lstm, sess


vocab, category, lstm, sess = load_model()


def evaluate_line(text):
    id_words = vocab.sentence_to_id(text)
    id_words = id_words[0: hps.num_timesteps]
    padding_num = hps.num_timesteps - len(id_words)
    id_words = id_words + [vocab.unk for _ in range(padding_num)]
    batch_x = [id_words] * hps.batch_size

    _, predict_label = sess.run(
        [lstm.train_op, lstm.y_pred],
        feed_dict={
            lstm.inputs: np.array(batch_x),
            lstm.outputs: np.array([0] * hps.batch_size),
            lstm.keep_prob: 1.0
        }
    )
    return category.id_to_category.get(predict_label[0])


if __name__ == '__main__':
    import time
    # 财经
    text = """交银货币清明假期前两日暂停申购和转换入全景网3月30日讯 交银施罗德基金周一公告称，公司旗下的交银施罗德货币市场证券投资基金将于2009年"清明"假期前两日暂停申购和转换入业务。公告表示，交银施罗德货币将于2009年4月2日、3日两天暂停办理基金的申购和转换入业务。转换出、赎回等其他业务以及公司管理的其他开放式基金的各项交易业务仍照常办理。自2009年4月7日起，所有销售网点恢复办理基金的正常申购和转换入业务。(全景网/雷鸣)"""
    t1 = time.time()
    label = evaluate_line(text=text)
    t2 = time.time()
    print(label)
    print('cost time: {0}ms'.format(t2 - t1))
