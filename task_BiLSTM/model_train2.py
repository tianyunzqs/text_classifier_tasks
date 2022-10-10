# -*- coding: utf-8 -*-
# @Time        : 2021/9/18 15:18
# @Author      : tianyunzqs
# @Description :

import os
import pickle
import json
import jieba
import numpy as np
from tqdm import tqdm
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
from bert4keras.snippets import DataGenerator, sequence_padding
from sklearn import metrics
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0002, lr_min=0.0, decay=0.0, warmup_epochs=0):
        self.num_passed_batchs = 0          # 一个计数器
        self.warmup_epochs = warmup_epochs
        self.lr = lr_base                   # learning_rate_base
        self.lr_min = lr_min                # 最小的起始学习率,此代码尚未实现
        self.decay = decay                  # 指数衰减率
        self.steps_per_epoch = 0            # 也是一个计数器

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch==0:
            # 防止跑验证集的时候呗更改了
            if self.params['steps'] is None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self,epoch,logs=None):
        # 用来输出学习率的,可以删除
        print("learning_rate:", K.get_value(self.model.optimizer.lr))


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
# x_train = [jieba.lcut(d[0]) for d in train_data]
# y_train = [int(d[1]) for d in train_data]
# x_dev = [jieba.lcut(d[0]) for d in dev_data]
# y_dev = [int(d[1]) for d in dev_data]

# 建立分词器
tokenizer = Tokenizer()  # 创建一个Tokenizer对象，将一个词转换为正整数
tokenizer.fit_on_texts([list(d[0]) for d in train_data])  # 将词编号，词频越大，编号越小
# x_train = tokenizer.texts_to_sequences(x_train)  # 将测试集列表中每个词转换为数字
# x_dev = tokenizer.texts_to_sequences(x_dev)  # 将测试集列表中每个词转换为数字
# train_data = [(x, y) for x, y in zip(x_train, y_train)]
# dev_data = [(x, y) for x, y in zip(x_dev, y_dev)]


def build_model():
    main_input = Input(shape=(MAX_LEN,), dtype='float64')
    # 嵌入层（使用预训练的词向量）
    embed = Embedding(input_dim=len(tokenizer.index_word) + 1, output_dim=128)(main_input)
    out = Bidirectional(LSTM(units=64))(embed)
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
    warm_up = WarmupExponentialDecay(lr_base=0.0002, decay=0.00002, warmup_epochs=2)
    model.fit_generator(train_data_generator.forfit(),
                        steps_per_epoch=len(train_data_generator),
                        epochs=50,
                        validation_data=dev_data_generator.forfit(),
                        validation_steps=len(dev_data_generator),
                        callbacks=[evaluator])
