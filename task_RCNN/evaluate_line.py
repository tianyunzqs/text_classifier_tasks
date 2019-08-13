# -*- coding: utf-8 -*-
# @Time        : 2019/8/8 15:31
# @Author      : tianyunzqs
# @Description : 

import json
import pickle

import jieba.posseg as pseg
import tensorflow as tf

from task_RCNN.textRCNN import TextRCNN


class Evaluation(object):
    def __init__(self):
        self.stopwords = self.load_stopwords(r'../sample_data/stopwords.txt')
        self.load_model()
        self.id_category = self.config['labels']

    @staticmethod
    def load_stopwords(path):
        stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
        return stopwords

    def load_model(self):
        with open("./models/vocab.model", 'rb') as f:
            self.vocab = pickle.loads(f.read())

        with open('./config_file.json', encoding="utf8") as f:
            self.config = json.load(f)

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.config['allow_soft_placement'],
                log_device_placement=self.config['log_device_placement'])
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.lstm = TextRCNN(vocab_size=self.vocab.size(),
                                     num_embedding_size=self.config['embedding_size'],
                                     num_lstm_nodes=self.config['num_lstm_nodes'],
                                     num_classes=len(self.config['labels']),
                                     representation_output_size=self.config['representation_output_size'])

                saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
                # Initialize all variables
                self.sess.run(tf.global_variables_initializer())

                ckpt = tf.train.get_checkpoint_state(self.config['checkpoint_dir'])
                saver.restore(self.sess, ckpt.model_checkpoint_path)

    def evaluate_line(self, text):
        segwords = " ".join([w.word for w in pseg.cut(text) if w.word not in self.stopwords])
        id_words = self.vocab.sentence_to_id(segwords)
        id_words = id_words[0: self.config['num_timesteps']]
        padding_num = self.config['num_timesteps'] - len(id_words)
        id_words = id_words + [self.vocab.unk for i in range(padding_num)]

        predictions = self.sess.run(
            [self.lstm.predictions],
            feed_dict={
                self.lstm.inputs: [id_words],
                self.lstm.keep_prob: 1.0
            })

        return self.config['labels'][str(predictions[0][0])]


eva = Evaluation()


def evaluate_line(text):
    return eva.evaluate_line(text)


if __name__ == '__main__':
    import time
    # 财经
    text = """交银货币清明假期前两日暂停申购和转换入全景网3月30日讯 交银施罗德基金周一公告称，公司旗下的交银施罗德货币市场证券投资基金将于2009年"清明"假期前两日暂停申购和转换入业务。公告表示，交银施罗德货币将于2009年4月2日、3日两天暂停办理基金的申购和转换入业务。转换出、赎回等其他业务以及公司管理的其他开放式基金的各项交易业务仍照常办理。自2009年4月7日起，所有销售网点恢复办理基金的正常申购和转换入业务。(全景网/雷鸣)"""
    t1 = time.time()
    label = evaluate_line(text=text)
    t2 = time.time()
    print(label)
    print('cost time: {0}ms'.format(t2 - t1))
