# -*- coding: utf-8 -*-
# @Time        : 2019/7/24 11:21
# @Author      : tianyunzqs
# @Description : 

import math

import numpy as np
import tensorflow as tf


class LSTM_Model(object):
    def __init__(self, hps, vocab_size, num_classes):
        """
        构建lstm
        :param hps: 参数对象
        :param vocab_size:  词表 长度
        :param num_classes:  分类数目
        :return:
        """
        num_timesteps = hps.num_timesteps  # 一个句子中有num_timesteps个词语
        batch_size = hps.batch_size

        # 设置两个 placeholder， 内容id 和 标签id
        self.inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
        self.outputs = tf.placeholder(tf.int32, (batch_size, ))

        # dropout keep_prob 表示要keep多少值，丢掉的是1-keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.global_step = tf.Variable(
            tf.zeros([], tf.int64),
            name='global_step',
            trainable=False)  # 可以保存 当前训练到了 哪一步，而且不训练

        # 随机的在均匀分布下初始化, 构建 embeding 层
        embeding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

        # 和 name_scope 作用是一样的，他可以定义指定 initializer
        # tf.name_scope() 和 tf.variable_scope() 的区别 参考：
        # https://www.cnblogs.com/adong7639/p/8136273.html
        with tf.variable_scope('embedding', initializer=embeding_initializer):
            # tf.varialble_scope() 一般 和 tf.get_variable() 进行配合
            # 构建一个 embedding 矩阵,shape 是 [词表的长度, 每个词的embeding长度 ]
            embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)

            # 每一个词，都要去embedding中查找自己的向量
            # [1, 10, 7] 是一个句子，根据 embedding 进行转化
            # 如： [1, 10, 7] -> [embedding[1], embedding[10], embedding[7]]
            embeding_inputs = tf.nn.embedding_lookup(embeddings, self.inputs)
            # 上句的输入： Tensor("embedding/embedding_lookup:0", shape=(100, 50, 16), dtype=float32)
            # 输出是一个三维矩阵，分别是：100 是 batch_size 大小，50 是 句子中的单词数量，16 为 embedding 向量长度

        # lstm 层

        # 输入层 大小 加上 输出层的大小，然后开方
        scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)

        with tf.variable_scope('lstm_nn', initializer=lstm_init):
            cells = []  # 保存两个lstm层
            # 循环这两层 lstm
            for i in range(hps.num_lstm_layers):
                # BasicLSTMCell类是最基本的LSTM循环神经网络单元。
                # 输入参数和BasicRNNCell差不多， 设置一层 的 lstm 神经元
                cell = tf.contrib.rnn.BasicLSTMCell(
                    hps.num_lstm_nodes[i],  # 每层的 节点个数
                    state_is_tuple=True     # 中间状态是否是一个元组
                )
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cells.append(cell)

            cell = tf.contrib.rnn.MultiRNNCell(cells)
            # 该方法的作用是：将两层的lstm 连到一起，比如：上层的输出是下层的输入
            # 此时的cell，已经是一个多层的lstm，但是可以当做单层的来操作，比较简单

            # 保存中间的一个隐含状态，隐含状态在初始化的时候初始化为0，也就是零矩阵
            initial_state = cell.zero_state(batch_size, tf.float32)

            # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1](最后一层的输出)]
            # _ 代表的是隐含状态
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, embeding_inputs, initial_state=initial_state
            )  # 现在的rnn_outputs 代表了每一步的输出

            # 获得最后一步的输出，也就是说，最后一个step的最后一层的输出
            last = rnn_outputs[:, -1, :]
            # print(last) Tensor("lstm_nn/strided_slice:0", shape=(100, 32), dtype=float32)

        # 将最后一层的输出 链接到一个全连接层上
        # 参考链接：https://www.w3cschool.cn/tensorflow_python/tensorflow_python-fy6t2o0o.html
        fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
        with tf.variable_scope('fc', initializer=fc_init):  # initializer 此范围内变量的默认初始值
            fc1 = tf.layers.dense(last,
                                  hps.num_fc_nodes,
                                  activation=tf.nn.relu,
                                  name='fc1')
            # 进行 dropout
            fc1_dropout = tf.nn.dropout(fc1, self.keep_prob)
            # 进行更换 参考：https://blog.csdn.net/UESTC_V/article/details/79121642

            logits = tf.layers.dense(fc1_dropout, num_classes, name='fc2')

        # 没有东西需要初始化，所以可以直接只用name_scope()
        with tf.name_scope('metrics'):
            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.outputs
            )

            # 该方法 做了三件事：1,labels 做 onehot，logits 计算softmax概率，3. 做交叉熵
            self.loss = tf.reduce_mean(softmax_loss)

            self.y_pred = tf.argmax(tf.nn.softmax(logits), 1)

            # 这里做了 巨大 修改，如果问题，优先检查这里！！！！！！
            # print(type(outputs), type(y_pred))
            correct_pred = tf.equal(self.outputs, tf.cast(self.y_pred, tf.int32))  # 这里也做了修改
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope('train_op'):
            tvars = tf.trainable_variables()  # 获取所有可以训练的变量
            for var in tvars:
                tf.logging.info('variable name: %s' % (var.name, ))  # 打印出所有可训练变量

            # 对 梯度进行 截断.
            # grads是截断之后的梯度
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),  # 在可训练的变量的梯度
                hps.clip_lstm_grads
            )  # 可以 获得 截断后的梯度

            optimizer = tf.train.AdamOptimizer(hps.learning_rate)  # 将每个梯度应用到每个变量上去
            self.train_op = optimizer.apply_gradients(
                zip(grads, tvars),  # 将 梯度和参数 绑定起来
                global_step=self.global_step  # 这个参数 等会儿，再好好研究一下
            )

    def eval_holdout(self, sess, dataset, batch_size):
        # 计算出 该数据集 有多少batch
        num_batches = dataset.num_samples() // batch_size  # // 整除 向下取整

        accuracy_vals, loss_vals = [], []
        for i in range(num_batches):
            batch_inputs, batch_labels = dataset.next_batch(batch_size)
            accuracy_val, loss_val = sess.run(
                [self.accuracy, self.loss],
                feed_dict={
                    self.inputs: batch_inputs,
                    self.outputs: batch_labels,
                    self.keep_prob: 1.0
                })
            accuracy_vals.append(accuracy_val)
            loss_vals.append(loss_val)

            return np.mean(accuracy_vals), np.mean(loss_vals)
