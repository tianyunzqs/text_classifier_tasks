# -*- coding: utf-8 -*-
# @Time        : 2019/3/13 18:14
# @Author      : tianyunzqs
# @Description : 在线预测

import os
import sys
import json

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
    model_dir = r'./output'
else:
    model_dir = './output'
    bert_dir = '/home/fhpt/zqs/alg_nlp/pre_train_model/chinese_L-12_H-768_A-12'


with open("./config_file", encoding="utf8") as f:
    config = json.load(f)

batch_size = 1
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
        input_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
        input_mask_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")

        bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
        total_loss, logits, trans, pred_ids = create_model(
            bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p,
            segment_ids=None, labels=0, num_labels=num_labels, use_one_hot_embeddings=False)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=True)


def evaluate_line(text):

    def convert(line):
        text_a = tokenization.convert_to_unicode(line)
        line_example = InputExample(guid='test_example', text_a=text_a, text_b=None, label=None)
        feature = convert_single_example(line_example, label_list, max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids], (batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids

    input_ids, input_mask, segment_ids = convert(text)

    feed_dict = {
        input_ids_p: input_ids,
        input_mask_p: input_mask,
    }
    pred_ids_result = sess.run([pred_ids], feed_dict)[0][0]
    predict_label = label_list[np.argmax(pred_ids_result)]
    return predict_label


if __name__ == '__main__':
    text = "宏达股份钼铜项目或走他乡 募投恐推倒重来 开工仪式刚刚过去四天,宏达股份(600331)最重要的募投项目就走进了“死胡同”。宏达股份7月5日发布的公告引述当地政府的说法:“什邡今后不再建设这个项目(钼铜多金属资源深加工综合利用项目)”,这意味着钼铜项目已经无缘在什邡市建设。不过,作为四川省重点项目相关公司股票走势,宏达股份也许并不会放弃总投资逾百亿元的钼铜项目。一位什邡市官员透露:“下一步将取决于企业行为,可能会在其他地方重新选址建设。”“娘家”难容身“之所以当初钼铜项目选择在什邡市建设,一是由于宏达股份是本地企业,二是什邡是5·12特大地震的重灾区,希望通过投资拉动本地经济建设”,什邡市委宣传部的一位官员道出了当初引入钼铜多金属资源深加工综合利用项目的初衷。不过,尽管当地政府是为了“在一片废墟上建成灾后美好新家园”,但是由于群众反对,7月3日政府门户网站“什邡之窗”登载了《什邡今后不再建设宏达钼铜项目》的信息。什邡市政府表示:“经市委、市政府研究决定,坚决维护群众的合法权益,鉴于部分群众因担心宏达钼铜项目建成后,会影响环境,危及身体健康,反应十分强烈,决定停止该项目建设,什邡今后不再建设这个项目。”资料显示,钼铜多金属资源深加工综合利用项目主要包括建设钼4万t/a装置、阴极铜40万t/a装置、伴生稀贵金属回收装置、冶炼烟气制硫酸装置等,总投资额101.88亿元。什邡市委宣传部人士表示:“市政府态度已经十分明确,接下来就取决于企业行为了。钼铜项目属国家产业政策鼓励类项目,也是四川省\"十二五规划\"优势特色产业重点项目,可能在其他地方重新选址建设。”作为什邡市的本地企业,宏达股份募投项目遭到“娘家”反对,大大出乎了企业的预料。宏达股份高管4日下午召开了长时间的内部会议,内部人士会后表示,上市公司目前针对钼铜项目如何建设还没有形成具体的方案。董秘王延俊接受中国证券报记者采访时表示:“现在谈论钼铜项目的问题太敏感了,宏达股份的任何表态都会引起市场的波动。如果有新的进展,我们会及时进行披露。”募投恐推倒重来从2010年11月,宏达股份公告开展钼铜多金属资源深加工综合利用项目,到2012年6月末项目开工奠基,前后历经一年半,期间包含了融资、矿产资源储备、环评等众多努力。随着什邡市“不再建”的表态,宏达股份如果希望继续这一投资巨大的募投项目,必须重新选址,前期的项目备案、安全审查、节能审查等工作将会推倒重来。如今摆在宏达股份面前的将是一系列难题。首先,大量的矿产储备如何消化。按照《铜冶炼行业准入条件》,自有矿山原料比例需达到25%以上。为了满足钼铜多金属资源深加工的需要,宏达股份一直在扩张自己的矿产储备。此前,宏达股份矿石原料计划中,40万吨铜金属中50%来自于国内关联方自有矿山和战略关联方。公司7月3日的公告也显示,宏达股份、西藏自治区地质矿产勘查开发局第五地质大队、四川宏达(集团)有限公司、成都沃美东龙投资有限公司签订合作投资协议,拟共同投资设立西藏宏达多龙矿业有限公司,对西藏改则县多龙矿区进行地质勘查开发。如果重新选址,在宏达股份启动新的钼铜项目前,已经储备的大量矿产储备如何处理将是难题。其次,业绩将经受考验。2011年,宏达股份实现净利润5252.74万元,摆脱了2010年巨亏的泥潭,不过之所以业绩翻身主要是由于报告期内公司转让成都置成地产股权实现投资收益7.32亿元。如果钼铜项目短期内难以推进,就宏达股份目前的主营业务来看,铅锌价格震荡下跌,拖累了上市公司的采矿业务;冶炼加工费持续走低、环保成本日益提高等因素,导致公司金属冶炼业务毛利率进一步下滑9.55%,至8.57%。 (来源:中国证券报-中证网)"
    text = "Ivy Bridge平台 【ASUS(华硕) 华硕 M2412系列,索尼 YA2系列 Ivy Bridge平台 笔记本电脑】产品搜索"
    print(evaluate_line(text=text))
