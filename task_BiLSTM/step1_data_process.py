# -*- coding: utf-8 -*-
# @Time        : 2019/6/4 14:44
# @Author      : tianyunzqs
# @Description : 

import codecs

import jieba
from tqdm import tqdm


def text2seg(in_path, save_path, vocab_path, is_training=False):
    vocab = dict()
    with codecs.open(in_path, 'r', encoding='utf-8') as f, codecs.open(save_path, 'w', encoding='utf-8') as fout:
        lines = f.readlines()
        for line in tqdm(lines):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            label, content = parts
            words = jieba.lcut(content)
            for word in words:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
            segs = ' '.join(words)
            fout.write(label + '\t' + segs + '\n')
    if is_training:
        with codecs.open(vocab_path, 'w', encoding='utf-8') as f_vocab:
            f_vocab.write('<UNK>\t100\n')
            for word, freq in vocab.items():
                f_vocab.write(word + '\t' + str(freq) + '\n')


if __name__ == '__main__':
    train_file = r'D:\alg_file\data\cnews\cnews.train.txt'
    train_save_file = r'D:\alg_file\data\cnews\cnews.train.seg.txt'
    vocab_save_path = r'D:\alg_file\data\cnews\cnews.vocab.txt'
    text2seg(train_file, train_save_file, vocab_save_path, is_training=True)

    val_file = r'D:\alg_file\data\cnews\cnews.val.txt'
    val_save_file = r'D:\alg_file\data\cnews\cnews.val.seg.txt'
    text2seg(val_file, val_save_file, '', is_training=False)

    test_file = r'D:\alg_file\data\cnews\cnews.test.txt'
    test_save_file = r'D:\alg_file\data\cnews\cnews.test.seg.txt'
    text2seg(test_file, test_save_file, '', is_training=False)
