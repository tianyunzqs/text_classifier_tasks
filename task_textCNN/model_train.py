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
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
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
            data.append((line_json['sentence'], line_json['label']))
    return data


MAX_LEN = 256
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
print(1)


def build_model():
    main_input = Input(shape=(MAX_LEN,), dtype='float64')
    # 嵌入层（使用预训练的词向量）
    embedder = Embedding(input_dim=len(tokenizer.index_word) + 1, output_dim=128)
    embed = embedder(main_input)
    # 卷积层和池化层，设置卷积核大小分别为3,4,5
    cnn_features = []
    for filter_size in [3, 4, 5]:
        # shape = [batch_size, maxlen-filter_size+1, embed_size]
        cnn = Conv1D(filters=128, kernel_size=filter_size)(embed)
        cnn = GlobalMaxPooling1D()(cnn)
        # cnn = MaxPooling1D(pool_size=MAX_LEN - filter_size + 1)(cnn)
        cnn_features.append(cnn)

    # 合并三个模型的输出向量
    cnn = concatenate(cnn_features, axis=-1)
    # cnn = Flatten()(cnn)
    drop = Dropout(0.5)(cnn)  # 在池化层到全连接层之前可以加上dropout防止过拟合
    main_output = Dense(num_classes, activation='softmax')(drop)
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
            batch_labels.append(int(label))
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=MAX_LEN)
                yield batch_token_ids, batch_labels
                batch_token_ids, batch_labels = [], []


def evaluate(model, data):
    X, Y = [], []
    for x, y in tqdm(data):
        x = tokenizer.texts_to_sequences([list(x)])[0]
        if len(x) > MAX_LEN:
            X.append(x[:MAX_LEN])
        else:
            X.append(x + [0] * (MAX_LEN - len(x)))
        Y.append(int(y))
    predict_Y = model.predict(np.array(X), batch_size=16)
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

    train_data_generator = TyDataGenerator(train_data, batch_size=16)
    dev_data_generator = TyDataGenerator(dev_data, batch_size=16)

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
