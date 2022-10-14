# -*- coding: utf-8 -*-
# @Time        : 2022/10/14 9:09
# @Author      : tianyunzqs
# @Description :

import os
import json
from tqdm import tqdm
from keras.layers import Lambda, Dense
from bert4keras.optimizers import Adam
from bert4keras.backend import K, keras, search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding
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


maxlen = 128
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
            batch_labels.append([int(label)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=0.1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp


# 转换数据集
train_generator = TyDataGenerator(train_data, batch_size)
dev_generator = TyDataGenerator(dev_data, batch_size)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
model.compile(
    loss=loss_with_gradient_penalty,
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
        epochs=20,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
