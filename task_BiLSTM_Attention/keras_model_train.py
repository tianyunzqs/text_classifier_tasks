# -*- coding: utf-8 -*-
# @Time        : 2022/10/9 18:28
# @Author      : tianyunzqs
# @Description :

import os
import pickle
import json
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras import initializers
from keras.callbacks import ReduceLROnPlateau, Callback
from bert4keras.snippets import DataGenerator, sequence_padding
from sklearn import metrics
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            data.append((line_json['sentence'], int(line_json['label'])))
    return data


batch_size = 32
MAX_LEN = 512
# 加载数据集
train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set([d[1] for d in train_data]))
dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
# 建立分词器
tokenizer = Tokenizer()  # 创建一个Tokenizer对象，将一个词转换为正整数
tokenizer.fit_on_texts([list(d[0]) for d in train_data])  # 将词编号，词频越大，编号越小


class TyAttentionLayer(Layer):
    """
    严格公式实现
    """
    def __init__(self,
                 attention_size=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        self.attention_size = attention_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(TyAttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.attention_size),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.attention_size),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.attention_size),
                                        initializer='glorot_normal')

        super(TyAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        q, k, v = inputs[:3]

        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)

        qk = K.batch_dot(qw, kw, [2, 2]) / (self.attention_size ** 0.5)
        qksf = K.softmax(qk, axis=1)
        o = K.batch_dot(qksf, vw, [1, 1])
        outputs = K.sum(o, axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.attention_size


class AttentionLayer(Layer):
    """
    网上版本
    """
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def build_model():
    main_input = Input(shape=(MAX_LEN,), dtype='float64')
    # 嵌入层（使用预训练的词向量）
    embed = Embedding(input_dim=len(tokenizer.index_word) + 1, output_dim=128)(main_input)
    out = Bidirectional(LSTM(units=64, return_sequences=True))(embed)
    # out = AttentionLayer(64)(out)
    out = TyAttentionLayer(128)([out, out, out])
    out = Dropout(0.3)(out)
    main_output = Dense(num_classes, activation='softmax')(out)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['sparse_categorical_accuracy'])
    return model


class TyDataGenerator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_labels = [], []
        for is_end, (text, label) in self.sample(random):
            token_ids = tokenizer.texts_to_sequences([list(text)])[0]
            batch_token_ids.append(token_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=MAX_LEN)
                yield batch_token_ids, batch_labels
                batch_token_ids, batch_labels = [], []


def evaluate(model, data):
    X, Y = [], []
    for x, y in data:
        x = tokenizer.texts_to_sequences([list(x)])[0]
        if len(x) > MAX_LEN:
            X.append(x[:MAX_LEN])
        else:
            X.append(x + [0] * (MAX_LEN - len(x)))
        Y.append(y)
    predict_Y = model.predict(np.array(X), batch_size=batch_size)
    y_predict = np.argmax(predict_Y, axis=1)
    print(metrics.classification_report(np.array(Y), y_predict, digits=4))
    val_acc = np.mean(np.equal(np.argmax(predict_Y, axis=1), np.array(Y)))
    return float(val_acc)


class Evaluator(Callback):
    def __init__(self, model, data, save_path):
        self.model = model
        self.data = data
        self.save_path = save_path
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.model, self.data)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.save_path)
        print('val_acc: %.5f, best_val_acc: %.5f' % (val_acc, self.best_val_acc))


if __name__ == '__main__':
    model = build_model()

    train_data_generator = TyDataGenerator(train_data, batch_size=batch_size)
    dev_data_generator = TyDataGenerator(dev_data, batch_size=batch_size)

    # call_reduce = ReduceLROnPlateau(monitor='val_acc',
    #                                 factor=0.95,
    #                                 patience=3,
    #                                 verbose=2,
    #                                 mode='auto',
    #                                 min_delta=0.01,
    #                                 cooldown=0,
    #                                 min_lr=0)
    model_path = './models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with open(os.path.join(model_path, 'tokenizer.plk'), 'wb') as f:
        pickle.dump(tokenizer, f)
    evaluator = Evaluator(model, dev_data, os.path.join(model_path, 'best_model.weights'))
    model.fit_generator(train_data_generator.forfit(),
                        steps_per_epoch=len(train_data_generator),
                        epochs=50,
                        validation_data=dev_data_generator.forfit(),
                        validation_steps=len(dev_data_generator),
                        callbacks=[evaluator])
