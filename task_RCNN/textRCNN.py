# -*- coding: utf-8 -*-
# @Time        : 2019/6/4 19:20
# @Author      : tianyunzqs
# @Description : 

import math

import tensorflow as tf


class TextRCNN(object):
    def __init__(self,
                 vocab_size, 
                 num_embedding_size,
                 num_lstm_nodes,
                 num_classes,
                 representation_output_size):
        self.inputs = tf.placeholder(tf.int32, [None, None], name='input')
        self.outputs = tf.placeholder(tf.int32, [None], name='output')
        # 随机失活剩下的神经单元  keep_prob = 1-dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
        '''embedding层'''
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):
            with tf.variable_scope('embedding', initializer=tf.contrib.layers.xavier_initializer()):
                embeddings = tf.get_variable('embedding', [vocab_size, num_embedding_size], tf.float32)
                embed_outputs = tf.nn.embedding_lookup(embeddings, self.inputs)  # change inputs to embedding
                embed_outputs_bak = embed_outputs
    
        '''BiLSTM层'''
        with tf.name_scope('bilstm_layer'):
            for idx, lstm_hidden_size in enumerate(num_lstm_nodes):
                with tf.name_scope('bilstm-{0}'.format(idx)):
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size, state_is_tuple=True),
                        output_keep_prob = self.keep_prob)
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=lstm_hidden_size, state_is_tuple=True),
                        output_keep_prob = self.keep_prob)
                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    # PS: ValueError: Variable bidirectional_rnn/fw/lstm_cell/kernel already exists, disallowed.
                    # Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
                    # 如果注释scope="bilstm" + str(idx)，则会出现以上错误
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                             cell_bw=lstm_bw_cell,
                                                                             inputs=embed_outputs,
                                                                             dtype=tf.float32,
                                                                             scope="bilstm" + str(idx))
                    embed_outputs = tf.concat(outputs, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        fw_output, bw_output = tf.split(embed_outputs, 2, -1)

        with tf.name_scope("context"):
            shape = [tf.shape(fw_output)[0], 1, tf.shape(fw_output)[2]]
            context_left = tf.concat([tf.zeros(shape), fw_output[:, :-1]], axis=1, name="context_left")
            context_right = tf.concat([bw_output[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("wordRepresentation"):
            word_representation = tf.concat([context_left, embed_outputs_bak, context_right], axis=2)
            word_size = num_lstm_nodes[-1] * 2 + num_embedding_size

        with tf.name_scope("text_representation"):
            text_w = tf.Variable(tf.random_uniform([word_size, representation_output_size], -1.0, 1.0), name="text_w")
            text_b = tf.Variable(tf.constant(0.1, shape=[representation_output_size]), name="text_b")

            # tf.einsum可以指定维度的消除运算
            text_representation = tf.tanh(tf.einsum('aij,jk->aik', word_representation, text_w) + text_b)

        # 做max-pool的操作，将时间步的维度消失
        rcnn_output = tf.reduce_max(text_representation, axis=1)

        '''output层'''
        with tf.name_scope('output_layer'):
            output_w = tf.get_variable(name='output_w', shape=[representation_output_size, num_classes])
            output_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='output_b')
            self.logits = tf.nn.xw_plus_b(rcnn_output, output_w, output_b, name="logits")

        '''计算损失函数'''
        with tf.name_scope('metrics'):
            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.outputs)
            # softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.outputs)
            self.loss = tf.reduce_mean(softmax_loss)

        '''计算准确率'''
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1, output_type=tf.int32, name="predictions")
            correct_pred = tf.equal(self.outputs, self.predictions)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
