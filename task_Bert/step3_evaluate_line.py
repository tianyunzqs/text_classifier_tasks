# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 在线预测

import os
import sys
import json
import time

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.abspath(__file__)))
import modeling, tokenization
from step2_train_model import create_model, InputFeatures, PaddingInputExample, InputExample


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=0,
        is_real_example=True)
    return feature


if os.name == 'nt':
    bert_dir = 'D:/alg_file/chinese_L-12_H-768_A-12'
    model_dir = r'./output_model'
else:
    model_dir = './output_model'
    bert_dir = '/home/fhpt/zqs/alg_nlp/pre_train_model/chinese_L-12_H-768_A-12'


with open("./config_file", encoding="utf8") as f:
    config = json.load(f)


max_seq_length = config['max_seq_length']
label_list = config['label_list']
num_labels = len(label_list)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True

graph = tf.Graph()
sess = tf.Session(config=gpu_config, graph=graph)

print('checkpoint path: {}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

with sess.as_default():
    with graph.as_default():
        print("going to restore checkpoint...")
        # sess.run(tf.global_variables_initializer())
        input_ids_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [None, max_seq_length], name="input_mask")

        bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
        total_loss, logits, trans, pred_ids = create_model(
            bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p,
            segment_ids=None, labels=0, num_labels=num_labels, use_one_hot_embeddings=False)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        print('checkpoint restored.')

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=True)


def evaluate_line(text):

    def convert(line):
        text_a = tokenization.convert_to_unicode(line)
        line_example = InputExample(guid='test_example', text_a=text_a, text_b=None, label=None)
        feature = convert_single_example(line_example, label_list, max_seq_length, tokenizer)
        return feature.input_ids, feature.input_mask, feature.segment_ids

    input_ids, input_mask, segment_ids = convert(text)

    feed_dict = {
        input_ids_p: input_ids,
        input_mask_p: input_mask,
    }
    pred_ids_result = sess.run(pred_ids, feed_dict)[0]
    predict_label = label_list[np.argmax(pred_ids_result)]
    return predict_label


def evaluate_batch(text_list):

    def convert(line):
        text_a = tokenization.convert_to_unicode(line)
        line_example = InputExample(guid='test_example', text_a=text_a, text_b=None, label=None)
        feature = convert_single_example(line_example, label_list, max_seq_length, tokenizer)
        return feature.input_ids, feature.input_mask, feature.segment_ids

    input_ids_list, input_mask_list = [], []
    for text in text_list:
        input_ids, input_mask, segment_ids = convert(text)
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)

    feed_dict = {
        input_ids_p: input_ids_list,
        input_mask_p: input_mask_list,
    }
    pred_ids_result = sess.run(pred_ids, feed_dict)
    predict_label = [label_list[np.argmax(res)] for res in pred_ids_result]
    return predict_label


if __name__ == '__main__':
    # 财经
    text = """交银货币清明假期前两日暂停申购和转换入全景网3月30日讯 交银施罗德基金周一公告称，公司旗下的交银施罗德货币市场证券投资基金将于2009年"清明"假期前两日暂停申购和转换入业务。公告表示，交银施罗德货币将于2009年4月2日、3日两天暂停办理基金的申购和转换入业务。转换出、赎回等其他业务以及公司管理的其他开放式基金的各项交易业务仍照常办理。自2009年4月7日起，所有销售网点恢复办理基金的正常申购和转换入业务。(全景网/雷鸣)"""
    t1 = time.time()
    label = evaluate_line(text=text)
    t2 = time.time()
    print(label)
    print('cost time: {0}s'.format(t2 - t1))
