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
from keras.losses import kullback_leibler_divergence as kld
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


def get_pre_model_path(model_name='bert'):
    """
    :param model_name: bert、bert-wwm、albert、roberta、roberta-wwm、nezha、t5
    :return:
    """
    if model_name == 'bert':
        _config_path = '/home/zqs/pre_train_models/chinese_L-12_H-768_A-12/bert_config.json'
        _checkpoint_path = '/home/zqs/pre_train_models/chinese_L-12_H-768_A-12/bert_model.ckpt'
        _dict_path = '/home/zqs/pre_train_models/chinese_L-12_H-768_A-12/vocab.txt'
        _model_name = 'bert'
    elif model_name == 'bert-wwm':
        _config_path = '/home/zqs/pre_train_models/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
        _checkpoint_path = '/home/zqs/pre_train_models/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        _dict_path = '/home/zqs/pre_train_models/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        _model_name = 'bert'
    elif model_name == 'albert':
        raise Exception("not support yet.")
    elif model_name == 'roberta':
        raise Exception("not support yet.")
    elif model_name == 'roberta-wwm':
        _config_path = '/home/zqs/pre_train_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
        _checkpoint_path = '/home/zqs/pre_train_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        _dict_path = '/home/zqs/pre_train_models/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
        _model_name = 'roberta'
    elif model_name == 'nezha':
        _config_path = '/home/zqs/pre_train_models/NEZHA-Base/bert_config.json'
        _checkpoint_path = '/home/zqs/pre_train_models/NEZHA-Base/model.ckpt-900000'
        _dict_path = '/home/zqs/pre_train_models/NEZHA-Base/vocab.txt'
        _model_name = 'nezha'
    elif model_name == 't5':
        _config_path = '/home/zqs/pre_train_models/chinese_t5_pegasus_base/bert_config.json'
        _checkpoint_path = '/home/zqs/pre_train_models/chinese_t5_pegasus_base/model.ckpt-900000'
        _dict_path = '/home/zqs/pre_train_models/chinese_t5_pegasus_base/vocab.txt'
        _model_name = 't5'
    else:
        raise Exception("not support yet.")
    return _config_path, _checkpoint_path, _dict_path, _model_name


MAX_LEN = 128
BATCH_SIZE = 32
# 加载数据集
train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set([d[1] for d in train_data]))
dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
config_path, checkpoint_path, dict_path, model_name = get_pre_model_path(model_name='bert')
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class TyDataGenerator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=MAX_LEN)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([int(label)])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


# 转换数据集
train_generator = TyDataGenerator(train_data, BATCH_SIZE)
dev_generator = TyDataGenerator(dev_data, BATCH_SIZE)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
    model=model_name,
    return_keras_model=False,
)
output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
model.compile(
    loss=crossentropy_with_rdrop,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)
model.load_weights('./models/best_model.weights')


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
            token_ids, segment_ids = tokenizer.encode(text, maxlen=MAX_LEN)
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
