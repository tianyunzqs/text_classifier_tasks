# -*- coding: utf-8 -*-
# @Time        : 2021/9/23 10:19
# @Author      : tianyunzqs
# @Description :

import os
import json
import jieba
from tqdm import tqdm
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from bert4keras.models import Transformer
from bert4keras.backend import keras, K
from bert4keras.layers import *
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TextTransformerKeras(Transformer):
    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act,  # FeedForward隐层的激活函数
        custom_position_ids=False,  # 是否自行传入位置id
        **kwargs  # 其余参数
    ):
        configs = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'hidden_act': hidden_act,
        }
        configs.update(kwargs)
        super(TextTransformerKeras, self).__init__(**configs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.custom_position_ids = custom_position_ids

    def get_inputs(self):
        """Transformer的输入是token_ids和position_ids
        """
        x_in = self.apply(
            layer=Input,
            shape=(MAX_LEN,),
            name='Input-Token'
        )
        inputs = [x_in]
        return inputs

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        x = self.apply(
            inputs=x,
            layer=PositionEmbedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            merge_mode='add',
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        return inputs


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            # if line_json['label'] not in ['70', '71']:
            #     continue
            data.append((line_json['sentence'], line_json['label']))
    return data


MAX_LEN = 256
batch_size = 32
# 加载数据集
train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
num_classes = len(set([d[1] for d in train_data]))
dev_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
x_train = [jieba.lcut(d[0]) for d in train_data]
# x_train = [list(d[0]) for d in train_data]
y_train = [int(d[1]) for d in train_data]
x_dev = [jieba.lcut(d[0]) for d in dev_data]
# x_dev = [list(d[0]) for d in dev_data]
y_dev = [int(d[1]) for d in dev_data]

# 建立分词器
tokenizer = Tokenizer()  # 创建一个Tokenizer对象，将一个词转换为正整数
tokenizer.fit_on_texts(x_train)  # 将词编号，词频越大，编号越小
x_train = tokenizer.texts_to_sequences(x_train)  # 将测试集列表中每个词转换为数字
x_dev = tokenizer.texts_to_sequences(x_dev)  # 将测试集列表中每个词转换为数字
train_data = [(x, y) for x, y in zip(x_train, y_train)]
dev_data = [(x, y) for x, y in zip(x_dev, y_dev)]

transformer = TextTransformerKeras(vocab_size=len(tokenizer.index_word),
                                   hidden_size=512,
                                   hidden_act="relu",
                                   num_hidden_layers=6,
                                   num_attention_heads=12,
                                   intermediate_size=256,
                                   dropout_rate=0.3)


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    return loss1


transformer.build()
# output = Reshape((-1, MAX_LEN * transformer.hidden_size))(transformer.model.output)
# output = Lambda(lambda x: x[:, 0])(transformer.model.output)
# output = Lambda(lambda x: K.mean(x, axis=1))(transformer.model.output)
output = GlobalAveragePooling1D()(transformer.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=transformer.initializer
)(output)

model = keras.models.Model(transformer.model.input, output)
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-4),
    metrics=['sparse_categorical_accuracy'],
)

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


def evaluate(data):
    X, Y = [], []
    for x, y in data:
        if len(x) > MAX_LEN:
            X.append(x[:MAX_LEN])
        else:
            X.append(x + [0] * (MAX_LEN - len(x)))
        Y.append(y)
    predict_Y = model.predict(np.array(X), batch_size=16)
    y_predict = np.argmax(predict_Y, axis=1)
    print(metrics.classification_report(np.array(Y), y_predict))
    val_acc = np.mean(np.equal(np.argmax(predict_Y, axis=1), np.array(Y)))
    return val_acc


def evaluate2(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        # y_true = y_true[:, 0]
        # total += len(y_true)
        total += 1
        right += (y_true == y_pred).sum()
    return right / total

# 转换数据集
train_data_generator = YqDataGenerator(train_data, batch_size)
dev_data_generator = YqDataGenerator(dev_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_path):
        self.best_val_acc = 0.
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(dev_data)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(os.path.join(self.model_path, 'best_model.weights'))
        print('val_acc: %.4f, best_val_acc: %.4f\n' % (val_acc, self.best_val_acc))


if __name__ == '__main__':
    evaluator = Evaluator(model_path='./models')
    model.fit_generator(train_data_generator.__iter__(),
                        steps_per_epoch=len(train_data_generator),
                        epochs=50,
                        validation_data=dev_data_generator.__iter__(),
                        validation_steps=len(dev_data_generator),
                        callbacks=[evaluator])

# else:
#     model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')

