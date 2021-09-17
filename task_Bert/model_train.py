# -*- coding: utf-8 -*-
# @Time        : 2021/9/16 16:13
# @Author      : tianyunzqs
# @Description :

import os
import json
from tqdm import tqdm
from keras.layers import Lambda, Dense
from bert4keras.optimizers import Adam
from bert4keras.backend import K, keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

maxlen = 128
batch_size = 16

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config_path = '/u01/zqs/pre_train_models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/u01/zqs/pre_train_models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/u01/zqs/pre_train_models/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            data.append((line_json['sentence'], line_json['label']))
    return data


# 加载数据集
train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set([d[1] for d in train_data]))
dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
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


# 转换数据集
train_generator = data_generator(train_data, batch_size)
dev_generator = data_generator(dev_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dropout_rate=0.3,
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
        epochs=50,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
