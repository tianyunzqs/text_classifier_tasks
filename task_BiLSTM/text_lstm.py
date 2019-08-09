# -*- coding: utf-8 -*-
# @Time        : 2019/6/4 19:20
# @Author      : tianyunzqs
# @Description : 

import math

import tensorflow as tf


class TextLSTM(object):
    def __init__(self, 
                 sequence_length,
                 batch_size,
                 vocab_size, 
                 num_embedding_size,
                 num_lstm_nodes,
                 num_fc_nodes,
                 num_classes):
        self.inputs = tf.placeholder(tf.int32, [batch_size, sequence_length])
        self.outputs = tf.placeholder(tf.int32, [batch_size, ])
        # 随机失活剩下的神经单元  keep_prob = 1-dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
        '''embedding层搭建'''
        # 定义初始化函数[-1,1]
        embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
        with tf.device('/cpu:0'):
            with tf.variable_scope('embedding', initializer=embedding_initializer):
                embeddings = tf.get_variable('embedding', [vocab_size, num_embedding_size], tf.float32)
                embed_inputs = tf.nn.embedding_lookup(embeddings, self.inputs)  # change inputs to embedding
    
        '''LSTM层搭建'''
        scale = (1.0 / math.sqrt(num_embedding_size + num_lstm_nodes[-1])) * 3.0
        lstm_init = tf.random_uniform_initializer(-scale, scale)
    
        def _generate_params_for_lstm_cell(x_size, h_size, bias_size):
            x_w = tf.get_variable('x_weights', x_size)
            h_w = tf.get_variable('h_weights', h_size)
            b = tf.get_variable('biases', bias_size, initializer=tf.constant_initializer(0.0))
            return x_w, h_w, b
    
        with tf.variable_scope('lstm_nn', initializer=lstm_init):
            with tf.variable_scope('inputs'):
                ix, ih, ib = _generate_params_for_lstm_cell(
                    x_size=[num_embedding_size, num_lstm_nodes[0]],
                    h_size=[num_lstm_nodes[0], num_lstm_nodes[0]],
                    bias_size=[1, num_lstm_nodes[0]]
                )
            with tf.variable_scope('outputs'):
                ox, oh, ob = _generate_params_for_lstm_cell(
                    x_size=[num_embedding_size, num_lstm_nodes[0]],
                    h_size=[num_lstm_nodes[0], num_lstm_nodes[0]],
                    bias_size=[1, num_lstm_nodes[0]]
                )
            with tf.variable_scope('forget'):
                fx, fh, fb = _generate_params_for_lstm_cell(
                    x_size=[num_embedding_size, num_lstm_nodes[0]],
                    h_size=[num_lstm_nodes[0], num_lstm_nodes[0]],
                    bias_size=[1, num_lstm_nodes[0]]
                )
            with tf.variable_scope('memory'):
                cx, ch, cb = _generate_params_for_lstm_cell(
                    x_size=[num_embedding_size, num_lstm_nodes[0]],
                    h_size=[num_lstm_nodes[0], num_lstm_nodes[0]],
                    bias_size=[1, num_lstm_nodes[0]]
                )
    
            state = tf.Variable(tf.zeros([batch_size, num_lstm_nodes[0]]), trainable=False)
            h = tf.Variable(tf.zeros([batch_size, num_lstm_nodes[0]]), trainable=False)
    
            for i in range(sequence_length):
                embed_input = embed_inputs[:, i, :]
                embed_input = tf.reshape(embed_input, [batch_size, num_embedding_size])
                forget_gate = tf.sigmoid(tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)
                input_gate = tf.sigmoid(tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
                output_gate = tf.sigmoid(tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)
                mid_state = tf.tanh(tf.matmul(embed_input, cx) + tf.matmul(h, ch) + cb)
                state = mid_state * input_gate + state * forget_gate
                h = output_gate * tf.tanh(state)
                last = h
            '''
            cells = []
            for i in range(num_lstm_layers):
                cell = tf.contrib.rnn.BasicLSTMCell(num_lstm_nodes[i], state_is_tuple = True)
                cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells)#Cell is 多层LSTM
    
            initial_state = cell.zero_state(None,tf.float32) #初始化隐藏状态为0
    
            #RNNoutput:
            #    一维：None
            #    二维：num_timesteps
            #    三维：lstm_outputs[-1]
    
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell,embed_inputs,initial_state=initial_state)
            last = rnn_outputs[:, -1, : ]
            '''
        '''FC层搭建'''
        fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
        with tf.variable_scope('fc', initializer=fc_init):
            fc1 = tf.layers.dense(last, num_fc_nodes, activation=tf.nn.relu, name='fc1')
            fc1_dropout = tf.contrib.layers.dropout(fc1, self.keep_prob)
            self.logits = tf.layers.dense(fc1_dropout, num_classes, name='fc2')
    
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
