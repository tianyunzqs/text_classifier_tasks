# -*- coding: utf-8 -*-
# @Time        : 2019/6/4 19:20
# @Author      : tianyunzqs
# @Description :

import tensorflow as tf


class TextLSTM(object):
    def __init__(self,
                 sequence_length,
                 vocab_size, 
                 num_embedding_size,
                 num_lstm_nodes,
                 num_classes):
        self.inputs = tf.placeholder(tf.int32, [None, None], name='input')
        self.outputs = tf.placeholder(tf.int32, [None], name='output')
        # 随机失活剩下的神经单元  keep_prob = 1-dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.sequence_length = sequence_length
        self.hidden_size_list = num_lstm_nodes
    
        '''embedding层'''
        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):
            with tf.variable_scope('embedding', initializer=tf.contrib.layers.xavier_initializer()):
                embeddings = tf.get_variable('embedding', [vocab_size, num_embedding_size], tf.float32)
                embed_outputs = tf.nn.embedding_lookup(embeddings, self.inputs)  # change inputs to embedding
    
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
        lstm_output = tf.split(embed_outputs, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope('attention_layer'):
            H = lstm_output[0] + lstm_output[1]
            attention_output = self._attention(H)

        attention_output_size = num_lstm_nodes[-1]

        '''output层'''
        with tf.name_scope('output_layer'):
            output_w = tf.get_variable(name='output_w', shape=[attention_output_size, num_classes])
            output_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='output_b')
            self.logits = tf.nn.xw_plus_b(attention_output, output_w, output_b, name="logits")

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

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hidden_size = self.hidden_size_list[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，M=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.sequence_length])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.sequence_length, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        # sequeezeR = tf.squeeze(r)  # 这句对于预测的时候，需要输入batch_size大小的样本
        sequeezeR = tf.reshape(r, [-1, hidden_size])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.keep_prob)

        return output
