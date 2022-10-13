# -*- coding: utf-8 -*-
# @Time        : 2021/9/16 16:13
# @Author      : tianyunzqs
# @Description :

import os
import json
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.layers import Lambda, Dense
from bert4keras.optimizers import Adam
from bert4keras.backend import K, keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)                 # tf2.x
    tf.compat.v1.random.set_random_seed(seed)  # tf1.x


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            data.append((line_json['sentence'], line_json['label']))
    return data


def load_pre_model(model_name='bert'):
    """
    :param model_name: bert、bert-wwm、albert、roberta、roberta-wwm、nezha、t5
    :return:
    """
    if model_name == 'bert':
        config_path = '/home/zqs/pre_train_models/chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = '/home/zqs/pre_train_models/chinese_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = '/home/zqs/pre_train_models/chinese_L-12_H-768_A-12/vocab.txt'
        _model = 'bert'
    elif model_name == 'bert-wwm':
        config_path = '/home/zqs/pre_train_models/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = '/home/zqs/pre_train_models/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = '/home/zqs/pre_train_models/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        _model = 'bert'
    elif model_name == 'albert':
        raise Exception("not support yet.")
    elif model_name == 'roberta':
        raise Exception("not support yet.")
    elif model_name == 'roberta-wwm':
        config_path = '/home/zqs/pre_train_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = '/home/zqs/pre_train_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = '/home/zqs/pre_train_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
        _model = 'roberta'
    elif model_name == 'nezha':
        config_path = '/home/zqs/pre_train_models/NEZHA-Base/bert_config.json'
        checkpoint_path = '/home/zqs/pre_train_models/NEZHA-Base/model.ckpt-900000'
        dict_path = '/home/zqs/pre_train_models/NEZHA-Base/vocab.txt'
        _model = 'nezha'
    elif model_name == 't5':
        config_path = '/home/zqs/pre_train_models/chinese_t5_pegasus_base/bert_config.json'
        checkpoint_path = '/home/zqs/pre_train_models/chinese_t5_pegasus_base/model.ckpt-900000'
        dict_path = '/home/zqs/pre_train_models/chinese_t5_pegasus_base/vocab.txt'
        _model = 't5'
    else:
        raise Exception("not support yet.")
    # 建立分词器
    pre_tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # 加载预训练模型
    pre_model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=_model,
        return_keras_model=False
    )
    return pre_tokenizer, pre_model


seed_tensorflow(42)
maxlen = 256
batch_size = 16
# 加载数据集
train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set([d[1] for d in train_data]))
dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
tokenizer, bert = load_pre_model(model_name='bert')


class TyDataGenerator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([int(float(label))])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def textcnn(inputs):
    # 选用3、4、5三个卷积核进行特征提取，最后拼接后输出用于分类。
    kernel_size = [3, 4, 5]
    cnn_features = []
    for size in kernel_size:
        cnn = keras.layers.Conv1D(filters=128, kernel_size=size)(inputs)  # shape=[batch_size,maxlen-2,256]
        cnn = keras.layers.GlobalMaxPooling1D()(cnn)  # shape=[batch_size,256]
        cnn_features.append(cnn)
    # 对kernel_size=3、4、5时提取的特征进行拼接
    output = keras.layers.concatenate(cnn_features, axis=-1)  # [batch_size,256*3]
    # 返回textcnn提取的特征结果
    return output


# 转换数据集
train_generator = TyDataGenerator(train_data, batch_size)
dev_generator = TyDataGenerator(dev_data, batch_size)

cls_features = Lambda(lambda x: x[:, 0])(bert.model.output)
# 去除第一个[cls]和最后一个[sep]，得到输入句子的embedding，用作textcnn的输入。
word_embedding = keras.layers.Lambda(lambda x: x[:, 1:-1], name='word_embedding')(bert.model.output)  # shape=[batch_size,maxlen-2,768]
# 将句子的embedding，输入textcnn，得到经由textcnn提取的特征。
cnn_features = textcnn(word_embedding)  # shape=[batch_size,cnn_output_dim]
# 将cls特征与textcnn特征进行拼接。
all_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)  # shape=[batch_size,cnn_output_dim+768]
# 应用dropout缓解过拟合的现象，rate一般在0.2-0.5。
all_features = keras.layers.Dropout(0.2)(all_features)  # shape=[batch_size,cnn_output_dim+768]
# 降维
output = keras.layers.Dense(units=256, activation='relu')(all_features)  # shape=[batch_size,256]

output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_path):
        self.best_val_acc = 0.
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(dev_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(os.path.join(self.model_path, 'best_model.weights'))
        print('val_acc: %.4f, best_val_acc: %.4f\n' % (val_acc, self.best_val_acc))


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    evaluator = Evaluator(model_path='./models')
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
