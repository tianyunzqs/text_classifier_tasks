# -*- coding: utf-8 -*-
# @Time        : 2019/6/3 15:05
# @Author      : tianyunzqs
# @Description : 

import os
import math

import numpy as np
import tensorflow as tf


def get_default_params():
    return tf.contrib.training.HParams(
        num_embedding_size=16,  #32                 #each Vec's length
        num_timesteps = 50, #600                    #LSTM步长
        num_lstm_nodes = [32, 32],  #[64,64]               #每一层的size32 ，2层
        num_lstm_layers = 2, #层数
        num_fc_nodes = 32,  #64      #全连接层神经单元数
        batch_size = 100,
        clip_lstm_grads = 1.0,#控制LSTM梯度大小
        learning_rate = 0.001,
        num_word_threshold = 10
    )


hps = get_default_params()

train_file = r'D:\alg_file\data\cnews\cnews.train.seg.txt'
val_file = r'D:\alg_file\data\cnews\cnews.val.seg.txt'
test_file = r'D:\alg_file\data\cnews\cnews.test.seg.txt'
vocab_file = r'D:\alg_file\data\cnews\cnews.vocab.txt'
category_file = r'D:\alg_file\data\cnews\cnews.category.txt'
output_folder = 'run_text_rnn'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

#test
print(hps.num_word_threshold)


# 词表封装模块
class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    # 读取文件每一行，赋值一个id
    def _read_dict(self, filename):
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                word, frequency = line.strip('\r\n').split('\t')
                frequency = int(frequency)
                if frequency < self._num_word_threshold:
                    continue
                idx = len(self._word_to_id)
                if word == '<UNK>':
                    self._unk = idx
                self._word_to_id[word] = idx

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) \
                    for cur_word in sentence.split()]
        return word_ids


# test
vocab = Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()
tf.logging.info('vocab_size: %d' % vocab_size)
print(vocab_size)

test_str = '的 在 了 是'
print(vocab.sentence_to_id(test_str))


# 类别封装模块
class CategeoryDict:
    def __init__(self, filename):
        self._categeory_to_id = {}
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            categeory = line.strip('\r\n')
            idx = len(self._categeory_to_id)
            self._categeory_to_id[categeory] = idx

    def categeory_to_id(self, categeory):
        if not categeory in self._categeory_to_id:
            raise Exception(
                "%s is not in our categeory list" % categeory)
        return self._categeory_to_id[categeory]

    def size(self):
        return len(self._categeory_to_id)


# test
categeory_vocab = CategeoryDict(category_file)
test_str = '娱乐'
num_classes = categeory_vocab.size()
tf.logging.info('label: %s,id: %d' % (test_str, categeory_vocab.categeory_to_id(test_str)))
tf.logging.info('num_classes: %d' % num_classes)


# 数据集封装模块
class TextDataSet:
    def __init__(self, filename, vocab, categeory_vocab, num_timesteps):
        self._vocab = vocab
        self._categeory_vocab = categeory_vocab
        self._num_timesteps = num_timesteps
        self._inputs = []  # matrxi
        self._outputs = []  # Vec
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        tf.logging.info('Loading data from %s', filename)
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        category_dict = {}
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            id_label = self._categeory_vocab.categeory_to_id(label)
            id_words = self._vocab.sentence_to_id(content)
            id_words = id_words[0:self._num_timesteps]
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)
        # 转换为numpy array
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator:end_indicator]
        batch_outputs = self._outputs[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs


# test
train_dataset = TextDataSet(train_file, vocab, categeory_vocab, hps.num_timesteps)
val_dataset = TextDataSet(val_file, vocab, categeory_vocab, hps.num_timesteps)
test_dataset = TextDataSet(test_file, vocab, categeory_vocab, hps.num_timesteps)

print(train_dataset.next_batch(2))
print(val_dataset.next_batch(2))
print(test_dataset.next_batch(2))


# 计算图构建模块
def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    # 随机失活剩下的神经单元  keep_prob = 1-dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # save training_step
    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    '''embedding层搭建'''
    # 定义初始化函数[-1,1]
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)  # change inputs to embedding

    '''LSTM层搭建'''
    scale = (1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1])) * 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)

    def _generate_params_for_lstm_cell(x_size, h_size, bias_size):
        x_w = tf.get_variable('x_weights', x_size)
        h_w = tf.get_variable('h_weights', h_size)
        b = tf.get_variable('biases', bias_size, initializer=tf.constant_initializer(0.0))
        return x_w, h_w, b

    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        with tf.variable_scope('inputs'):
            ix, ih, ib = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        with tf.variable_scope('outputs'):
            ox, oh, ob = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        with tf.variable_scope('forget'):
            fx, fh, fb = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )
        with tf.variable_scope('memory'):
            cx, ch, cb = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]]
            )

        state = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes[0]]), trainable=False)
        h = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes[0]]), trainable=False)

        for i in range(num_timesteps):
            embed_input = embed_inputs[:, i, :]
            embed_input = tf.reshape(embed_input, [batch_size, hps.num_embedding_size])
            forget_gate = tf.sigmoid(tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)
            input_gate = tf.sigmoid(tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
            output_gate = tf.sigmoid(tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)
            mid_state = tf.tanh(tf.matmul(embed_input, cx) + tf.matmul(h, ch) + cb)
            state = mid_state * input_gate + state * forget_gate
            h = output_gate * tf.tanh(state)
            last = h
        '''
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(hps.num_lstm_nodes[i], state_is_tuple = True)
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)#Cell is 多层LSTM

        initial_state = cell.zero_state(batch_size,tf.float32) #初始化隐藏状态为0

        #RNNoutput:
        #    一维：batch_size
        #    二维：num_timesteps
        #    三维：lstm_outputs[-1]

        rnn_outputs, _ = tf.nn.dynamic_rnn(cell,embed_inputs,initial_state=initial_state)
        last = rnn_outputs[:, -1, : ]
        '''
    '''FC层搭建'''
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last, hps.num_fc_nodes, activation=tf.nn.relu, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout, num_classes, name='fc2')

    '''计算损失函数'''
    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    '''计算图构建'''
    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()  # 获得所有训练变量
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), hps.clip_lstm_grads)  # 梯度截断
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return ((inputs, outputs, keep_prob), (loss, accuary), (train_op, global_step))


placeholders, metrics, others = create_model(hps, vocab_size, num_classes)
inputs, outputs, keep_prob = placeholders
loss, accuary = metrics
train_op, global_step = others


# 训练流程模块
init_op = tf.global_variables_initializer()
train_keep_prob_value = 0.8
test_keep_prob_value = 1.0

num_train_steps = 10000

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size)
        _, loss_val, accuary_val, global_step_val = sess.run([train_op, loss, accuary, global_step],
                                                             feed_dict={
                                                                 inputs: batch_inputs,
                                                                 outputs: batch_labels,
                                                                 keep_prob: train_keep_prob_value
                                                             })
        if global_step_val % 20 == 0:
            tf.logging.info("Step: %5d, loss: %3.3f, accuary: %3.3f" % (global_step_val, loss_val, accuary_val))
            print("Step: %5d, loss: %3.3f, accuary: %3.3f" % (global_step_val, loss_val, accuary_val))
