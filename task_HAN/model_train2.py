# -*- coding: utf-8 -*-
# @Time        : 2021/9/27 16:11
# @Author      : tianyunzqs
# @Description :

import os
import re
import json
import pickle
import jieba
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
from sklearn import metrics
from bert4keras.snippets import sequence_padding
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# class defining the custom attention layer
class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # return mask
        return None

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def load_data(path):
    x_data, y_data, texts = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            # if i > 50:
            #     break
            line_json = json.loads(line.strip())
            content, label = line_json['sentence'], line_json['label']
            sentences = re.split(r'[。；;！!？?]', content.strip())
            sentence_data = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                words = jieba.lcut(sentence.strip())
                texts.extend(words)
                sentence_data.append(words)
            x_data.append(sentence_data)
            y_data.append(int(label))
    return x_data, y_data, texts


MAX_LEN = 256
batch_size = 16
embedding_dim = 128
# 加载数据集
x_train, y_train, texts = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set(y_train))
x_dev, y_dev, _ = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))

# 建立分词器
tokenizer = Tokenizer()  # 创建一个Tokenizer对象，将一个词转换为正整数
tokenizer.fit_on_texts(texts)  # 将词编号，词频越大，编号越小
x_train = [tokenizer.texts_to_sequences(d) for d in x_train]  # 将测试集列表中每个词转换为数字
x_dev = [tokenizer.texts_to_sequences(d) for d in x_dev]  # 将测试集列表中每个词转换为数字
train_data = [(x, y) for x, y in zip(x_train, y_train)]
dev_data = [(x, y) for x, y in zip(x_dev, y_dev)]


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
            max_sentence_len, max_word_len = 0, 0
            for i in idxs:
                x, y = self.data[i]
                X.append(x[:])
                Y.append(y)
                max_sentence_len = max(max_sentence_len, len(x))
                max_word_len = max(max_word_len, max([len(xx) for xx in x]))
                if len(X) == self.batch_size or i == idxs[-1]:
                    for k in range(len(X)):
                        X[k].extend([[0] * max_word_len] * (max_sentence_len - len(X[k])))
                        X[k] = sequence_padding(X[k], length=max_word_len)
                    yield np.array(X), np.array(Y)
                    X, Y = [], []
                    max_sentence_len, max_word_len = 0, 0


def evaluate(model, data):
    y_true, y_pred = [], []
    for x, y in data:
        y_true.append(y)
        predict_Y = model.predict(np.array([sequence_padding(x)]))
        y_predict = np.argmax(predict_Y, axis=1)
        y_pred.extend(y_predict)
    print(metrics.classification_report(y_true, y_pred))
    val_acc = np.mean(np.equal(y_pred, y_true))
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


train_data_generator = YqDataGenerator(train_data, batch_size=batch_size)
dev_data_generator = YqDataGenerator(dev_data, batch_size=batch_size)

# sentence_input = Input(shape=(maxlen,), dtype='int32')
sentence_input = Input(shape=(None,), dtype='int32')
embedded_sequences = Embedding(input_dim=len(tokenizer.index_word) + 1, output_dim=embedding_dim,
                               trainable=True, mask_zero=True)(sentence_input)
lstm_word = Bidirectional(GRU(64, return_sequences=True))(embedded_sequences)
lstm_word = Dropout(0.2)(lstm_word)
attn_word = HierarchicalAttentionNetwork(128)(lstm_word)
sentenceEncoder = Model(sentence_input, attn_word)

# review_input = Input(shape=(max_sentences, maxlen), dtype='int32')
review_input = Input(shape=(None, None,), dtype='int32')
review_encoder = TimeDistributed(sentenceEncoder)(review_input)
review_encoder = Dropout(0.3)(review_encoder)
lstm_sentence = Bidirectional(GRU(64, return_sequences=True))(review_encoder)
attn_sentence = HierarchicalAttentionNetwork(128)(lstm_sentence)
attn_sentence = Dropout(0.3)(attn_sentence)
preds = Dense(num_classes, activation='softmax')(attn_sentence)
model = Model(review_input, preds)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=1e-3),
              metrics=['sparse_categorical_accuracy'])
# model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=10, batch_size=64)

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
