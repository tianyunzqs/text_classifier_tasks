# -*- coding: utf-8 -*-
# @Time        : 2019/6/3 14:14
# @Author      : tianyunzqs
# @Description :

import numpy as np


class Vocab(object):
    """
    词表封装模块
    """
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    # 读取文件每一行，赋值一个id
    def _read_dict(self, filename):
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                word, frequency = line.strip('\r\n').split('\t')
                frequency = int(frequency)
                if frequency < self._num_word_threshold:
                    continue
                idx = len(self._word_to_id)
                if word == '<UNK>':
                    self._unk = idx
                self._word_to_id[word] = idx

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) \
                    for cur_word in sentence.split()]
        return word_ids


class CategeoryDict(object):
    """
    类别封装模块
    """
    def __init__(self, filename):
        self._categeory_to_id = {}
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            categeory = line.strip('\r\n')
            idx = len(self._categeory_to_id)
            self._categeory_to_id[categeory] = idx

    def categeory_to_id(self, categeory):
        if not categeory in self._categeory_to_id:
            raise Exception(
                "%s is not in our categeory list" % categeory)
        return self._categeory_to_id[categeory]

    def size(self):
        return len(self._categeory_to_id)


class TextDataSet:
    """
    数据集封装模块
    """
    def __init__(self, filename, vocab, categeory_vocab, num_timesteps):
        self._vocab = vocab
        self._categeory_vocab = categeory_vocab
        self._num_timesteps = num_timesteps
        self._inputs = []  # matrxi
        self._outputs = []  # Vec
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            id_label = self._categeory_vocab.categeory_to_id(label)
            id_words = self._vocab.sentence_to_id(content)
            id_words = id_words[0:self._num_timesteps]
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)
        # 转换为numpy array
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        self._random_shuffle()

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
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_inputs = self._inputs[self._indicator:end_indicator]
        batch_outputs = self._outputs[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

    def batch_iter(self, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(self._inputs)
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(data_size)
                shuffled_inputs = self._inputs[shuffle_indices]
                shuffled_outputs = self._outputs[shuffle_indices]
            else:
                shuffled_inputs = self._inputs
                shuffled_outputs = self._outputs

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = (batch_num + 1) * batch_size
                if end_index <= data_size:
                # end_index = min((batch_num + 1) * batch_size, data_size)
                    yield epoch + 1, shuffled_inputs[start_index:end_index], shuffled_outputs[start_index:end_index]


def compute_p_r_f(y_true, y_pred):
    assert len(y_true) == len(y_pred), 'the length of y_true and y_pred is not equal.'
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum(np.equal(y_true, y_pred))
    p = tp / len(y_pred)
    r = tp / len(y_true)
    f = 2 * p * r / (p + r)
    return p, r, f
