# -*- coding: utf-8 -*-
# @Time        : 2019/7/24 11:02
# @Author      : tianyunzqs
# @Description :

import os
import sys


import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_LSTM_inbuild.data_helper import Vocab, CategoryDict, TextDataSet
from task_LSTM_inbuild.text_lstm import LSTM_Model

# 打印出 log
tf.logging.set_verbosity(tf.logging.INFO)


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

# 设置文件路径
train_file = 'D:/alg_file/data/cnews/cnews.train.seg.txt'
val_file = 'D:/alg_file/data/cnews/cnews.val.seg.txt'
test_file = 'D:/alg_file/data/cnews/cnews.test.seg.txt'
vocab_file = 'D:/alg_file/data/cnews/cnews.vocab.txt'
category_file = 'D:/alg_file/data/cnews/cnews.category.txt'

# 获得词表对象
vocab = Vocab(vocab_file, hps.num_word_threshold)
# 词表长度
vocab_size = vocab.size()


# 获得类别表对象
category_vocab = CategoryDict(category_file)
# 类别总数
num_classes = category_vocab.size()


# 得到三个文本对象，当中都包含了 input 和 label
train_dataset = TextDataSet(train_file, vocab, category_vocab, hps.num_timesteps)
val_dataset = TextDataSet(val_file, vocab, category_vocab, hps.num_timesteps)
test_dataset = TextDataSet(test_file, vocab, category_vocab, hps.num_timesteps)

lstm = LSTM_Model(hps, vocab_size, num_classes)

train_keep_prob_value = 0.8
num_train_steps = 100000
best_val_accuracy = 0

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "lstm_model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size)

        loss_val, accuracy_val, _, global_step_val = sess.run(
            [lstm.loss, lstm.accuracy, lstm.train_op, lstm.global_step],
            feed_dict={
                lstm.inputs: batch_inputs,
                lstm.outputs: batch_labels,
                lstm.keep_prob:train_keep_prob_value
            }
        )

        if global_step_val % 200 == 0:
            tf.logging.info('Step: %5d, loss: %3.3f, accuracy: %3.3f'%(global_step_val, loss_val, accuracy_val))

        if global_step_val % 1000 == 0:
            validdata_accuracy, validdata_loss = lstm.eval_holdout(sess, val_dataset, hps.batch_size)
            if validdata_accuracy > best_val_accuracy:
                best_val_accuracy = validdata_accuracy
                path = saver.save(sess, checkpoint_prefix, global_step=global_step_val)
                print("Saved model checkpoints to {}\n".format(path))
            testdata_accuracy, testdata_loss = lstm.eval_holdout(sess, test_dataset, hps.batch_size)
            tf.logging.info(' valid_data Step: %5d, loss: %3.3f, accuracy: %3.5f'
                            % (global_step_val, validdata_loss, validdata_accuracy))
            tf.logging.info(' test_data Step: %5d, loss: %3.3f, accuracy: %3.5f'
                            % (global_step_val, testdata_loss, testdata_accuracy))
