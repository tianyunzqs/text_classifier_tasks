# -*- coding: utf-8 -*-
# @Time        : 2019/7/24 11:12
# @Author      : tianyunzqs
# @Description : 

import numpy as np


class Vocab(object):
    def __init__(self, filename, num_word_threahold):
        # 每一个词，给她一个id，另外还要统计词频。
        # ps：前面带下划线的为私有成员
        self._word_to_id = {}
        self._unk = -1  # 先给 unk 赋值一个 负值，然后根据实际情况在赋值
        self._num_word_theshold = num_word_threahold  # 低于　这个值　就忽略掉该词
        self._read_dict(filename)  # 读词表方法

    def _read_dict(self, filename):
        """
        读这个词表
        :param filename: 路径
        :return: 
        """
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\n').split('\t')
            word = word  # 获得　单词
            frequency = int(frequency)  # 获得　频率
            if frequency < self._num_word_theshold:
                continue  # 门限过滤一下
            idx = len(self._word_to_id)  # 这里使用了一个id递增的小技巧
            if word == '<UNK>':          # 如果是空格，就把上一个id号给它
                # 如果是 unk的话， 就特殊处理一下
                self._unk = idx
            self._word_to_id[word] = idx
            # 如果 word 存在，就把 idx 当做值，将其绑定到一起
            # 如果 word 在词表中不存在，就把nuk的值赋予它

    def word_to_id(self, word):
        """
        为单词分配id值
        :param word: 单词
        :return:
        """
        # 字典.get() 如果有值，返回值；无值，返回默认值（就是第二个参数）
        return self._word_to_id.get(word, self._unk)

    def sentence_to_id(self, sentence):
        """
        将句子 转换成 id 向量
        :param sentence: 要输入的句子（分词后的句子）
        :return:
        """
        # 单条句子的id vector
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split(' ')]
        # cur_word 有可能不存在，需要使用函数进行过滤一下
        return word_ids

    # 定义几个 访问私有成员属性的方法
    # Python内置的 @ property装饰器就是负责把一个方法变成属性调用的
    @ property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)


class CategoryDict(object):
    """
    和 词表的 方法 几乎一样
    """
    def __init__(self, filename):
        self._category_to_id = {}
        self.id_to_category = {}
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx
        self.id_to_category = {v: k for k, v in self._category_to_id.items()}

    def size(self):
        return len(self._category_to_id)

    def category_to_id(self, category):
        if category not in self._category_to_id:
            raise Exception('%s is not in our category list' % category)
        return self._category_to_id[category]


class TextDataSet(object):
    """
    数据集 封装
    功能： 1、将数据集向量化。2、返回batch
    """
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        """
        封装数据集
        :param filename: 可以是训练数据集、测试数据集、验证数据集等
        :param vocab: 词表 对象
        :param category_vocab: 类别 对象
        :param num_timesteps: 步长 （sentence的总长度）
        """
        # 将　各个对象　赋值
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps

        # matrix
        self._inputs = list()
        # vector
        self._outputs = list()
        # batch 起始点
        self._indicator = 0

        # 将文本数据解析成matrix
        self._parse_file(filename)

    def _parse_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            label, content = line.strip('\n').split('\t')

            # 得到一个 label 的 id
            id_label = self._category_vocab.category_to_id(label)
            # 得到一个 vector
            id_words = self._vocab.sentence_to_id(content)

            # 需要在每一个minibatch上进行对齐，对 word 进行 对齐 操作
            # 如果超出了界限，就 截断， 如果 不足，就 填充
            id_words = id_words[0: self._num_timesteps]  # 超过了就截断
            # 低于 num_timesteps 就填充,也就是说，上一句和下面两句 可以完全并列写，神奇！！
            # 这里的编码方式感觉很巧妙！！！
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for _ in range(padding_num)]

            self._inputs.append(id_words)
            self._outputs.append(id_label)

        # 转变为 numpy 类型
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        # 对数据进行随机化
        self._random_shuffle()
        self._num_sample = len(self._inputs)

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception('batch_size: %d is too large' % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

    def num_samples(self):
        return self._num_sample
