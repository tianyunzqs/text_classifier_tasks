# -*- coding: utf-8 -*-
# @Time        : 2021/9/26 18:22
# @Author      : tianyunzqs
# @Description :

import os
import pickle
import json
import jieba
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
from sklearn import metrics
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            data.append((line_json['sentence'], line_json['label']))
    return data


MAX_LEN = 256
batch_size = 64
# 加载数据集
train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set([d[1] for d in train_data]))
dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
x_train = [jieba.lcut(d[0]) for d in train_data]
y_train = [int(d[1]) for d in train_data]
x_dev = [jieba.lcut(d[0]) for d in dev_data]
y_dev = [int(d[1]) for d in dev_data]

# 建立分词器
tokenizer = Tokenizer()  # 创建一个Tokenizer对象，将一个词转换为正整数
tokenizer.fit_on_texts([jieba.lcut(d[0]) for d in train_data])  # 将词编号，词频越大，编号越小
x_train = tokenizer.texts_to_sequences(x_train)  # 将测试集列表中每个词转换为数字
x_dev = tokenizer.texts_to_sequences(x_dev)  # 将测试集列表中每个词转换为数字
train_data = [(x, y) for x, y in zip(x_train, y_train)]
dev_data = [(x, y) for x, y in zip(x_dev, y_dev)]


def build_model():
    main_input = Input(shape=(MAX_LEN,), dtype='float64')
    # 嵌入层（使用预训练的词向量）
    embedder = Embedding(input_dim=len(tokenizer.index_word) + 1, output_dim=128)
    embed = embedder(main_input)
    embed_back = embed
    embed = Dropout(0.3)(embed)

    lstm_nodes = [128]
    for hidden_size in lstm_nodes[:-1]:
        embed = Bidirectional(LSTM(hidden_size, return_sequences=True))(embed)
    lstm_fw, lstm_bw = Bidirectional(LSTM(lstm_nodes[-1], return_sequences=True), merge_mode=None)(embed)

    out = concatenate([lstm_fw, embed_back, lstm_bw], axis=-1)
    out = Dense(256, activation='tanh')(out)
    out = GlobalMaxPooling1D()(out)
    out = Dropout(0.3)(out)
    main_output = Dense(num_classes, activation='softmax')(out)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['sparse_categorical_accuracy'])
    return model


class YqDataGenerator(object):
    def __init__(self, data,  batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, shuffle=False):
        while True:
            idxs = list(range(len(self.data)))
            if shuffle:
                np.random.shuffle(idxs)
            X, Y = [], []
            for i in idxs:
                x, y = self.data[i]
                if len(x) > MAX_LEN:
                    X.append(x[:MAX_LEN])
                else:
                    X.append(x + [0] * (MAX_LEN - len(x)))
                Y.append(y)
                if len(X) == self.batch_size or i == idxs[-1]:
                    yield np.array(X), np.array(Y)
                    X, Y = [], []


def evaluate(model, data):
    X, Y = [], []
    for x, y in data:
        if len(x) > MAX_LEN:
            X.append(x[:MAX_LEN])
        else:
            X.append(x + [0] * (MAX_LEN - len(x)))
        Y.append(y)
    predict_Y = model.predict(np.array(X), batch_size=batch_size)
    y_predict = np.argmax(predict_Y, axis=1)
    print(metrics.classification_report(np.array(Y), y_predict))
    val_acc = np.mean(np.equal(np.argmax(predict_Y, axis=1), np.array(Y)))
    return val_acc


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

    train_data_generator = YqDataGenerator(train_data, batch_size=batch_size)
    dev_data_generator = YqDataGenerator(dev_data, batch_size=batch_size)

    model_path = './models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with open(os.path.join(model_path, 'tokenizer.plk'), 'wb') as f:
        pickle.dump(tokenizer, f)
    evaluator = Evaluator(model, dev_data, os.path.join(model_path, 'best_model.weights'))
    model.fit_generator(train_data_generator.__iter__(),
                        steps_per_epoch=len(train_data_generator),
                        epochs=50,
                        validation_data=dev_data_generator.__iter__(),
                        validation_steps=len(dev_data_generator),
                        callbacks=[evaluator])
