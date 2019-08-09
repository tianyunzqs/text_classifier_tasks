# -*- coding: utf-8 -*-
# @Time        : 2019/6/3 15:05
# @Author      : tianyunzqs
# @Description :

import os
import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn import metrics

from task_BiLSTM_inbuild.data_helper import Vocab, CategeoryDict, TextDataSet, compute_p_r_f
from task_BiLSTM_inbuild.text_lstm import TextLSTM


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 16, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("num_timesteps", 50, "LSTM步长 (default: 50)")
tf.flags.DEFINE_list("num_lstm_nodes", [32, 32], "每一层的size32 ，2层")
tf.flags.DEFINE_integer("num_lstm_layers", 2, "层数")
tf.flags.DEFINE_integer("num_fc_nodes", 32, "全连接层神经单元数")
tf.flags.DEFINE_float("clip_lstm_grads", 1.0, "控制LSTM梯度大小")
tf.flags.DEFINE_float("learning_rate", 32, "学习率")
tf.flags.DEFINE_integer("num_word_threshold", 10, "阈值")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("config_file", "./config_file.json", "config file for model")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoints", "config file for model")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

train_file = r'D:\alg_file\data\cnews\cnews.train.seg.txt'
val_file = r'D:\alg_file\data\cnews\cnews.val.seg.txt'
test_file = r'D:\alg_file\data\cnews\cnews.test.seg.txt'
vocab_file = r'D:\alg_file\data\cnews\cnews.vocab.txt'
category_file = r'D:\alg_file\data\cnews\cnews.category.txt'


vocab = Vocab(vocab_file, FLAGS.num_word_threshold)
vocab_size = vocab.size()
tf.logging.info('vocab_size: %d' % vocab_size)

categeory_vocab = CategeoryDict(category_file)
num_classes = categeory_vocab.size()
tf.logging.info('num_classes: %d' % num_classes)


train_dataset = TextDataSet(train_file, vocab, categeory_vocab, FLAGS.num_timesteps)
val_dataset = TextDataSet(val_file, vocab, categeory_vocab, FLAGS.num_timesteps)
test_dataset = TextDataSet(test_file, vocab, categeory_vocab, FLAGS.num_timesteps)


def save_config(vocab, categeory):
    if not os.path.exists(r'./models'):
        os.makedirs(r'./models')
    with open(r'./models/vocab.model', 'wb') as f:
        pickle.dump(vocab, f)

    config = dict()
    # Model Hyperparameters
    config["embedding_size"] = FLAGS.embedding_size
    config["num_timesteps"] = FLAGS.num_timesteps
    config["num_lstm_nodes"] = FLAGS.num_lstm_nodes
    config["num_lstm_layers"] = FLAGS.num_lstm_layers
    config["num_fc_nodes"] = FLAGS.num_fc_nodes

    # Training parameters
    config["batch_size"] = FLAGS.batch_size
    config["config_file"] = FLAGS.config_file
    config["checkpoint_dir"] = FLAGS.checkpoint_dir
    # Misc Parameters
    config["allow_soft_placement"] = FLAGS.allow_soft_placement
    config["log_device_placement"] = FLAGS.log_device_placement

    config['labels'] = {v: k for k, v in categeory._categeory_to_id.items()}

    with open(FLAGS.config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)


save_config(vocab, categeory_vocab)


lstm = TextLSTM(vocab_size=vocab_size,
                num_embedding_size=FLAGS.embedding_size,
                num_lstm_nodes=FLAGS.num_lstm_nodes,
                num_classes=num_classes)

# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(lstm.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# add checkpoint
checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "model")
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

best_f = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练集
    for epoch, batch_inputs, batch_labels in train_dataset.batch_iter(FLAGS.batch_size, FLAGS.num_epochs):
        _, loss_val, accuracy_val, current_step = sess.run(
            [train_op, lstm.loss, lstm.accuracy, global_step],
            feed_dict={
                lstm.inputs: batch_inputs,
                lstm.outputs: batch_labels,
                lstm.keep_prob: 0.8
            })
        if current_step % 20 == 0:
            tf.logging.info("Epoch: %d, Step: %5d, loss: %3.3f, accuary: %3.3f"
                            % (epoch, current_step, loss_val, accuracy_val))

        if current_step % FLAGS.evaluate_every == 0:
            # 验证集
            val_labels_pred = val_labels_true = np.array([])
            for _, val_batch_inputs, val_batch_labels in val_dataset.batch_iter(FLAGS.batch_size, 1):
                _, val_predictions = sess.run(
                    [train_op, lstm.predictions],
                    feed_dict={
                        lstm.inputs: val_batch_inputs,
                        lstm.outputs: val_batch_labels,
                        lstm.keep_prob: 1.0
                    })
                val_labels_pred = np.append(val_labels_pred, val_predictions)
                val_labels_true = np.append(val_labels_true, val_batch_labels)

            classify_report = metrics.classification_report(val_labels_true, val_labels_pred)
            precision, recall, f_score = compute_p_r_f(val_labels_true, val_labels_pred)

            tf.logging.info("\nEvaluation: \n%s" % (classify_report, ))
            tf.logging.info("\nEvaluation: precision: %3.3f, recall: %3.3f, f_score: %3.3f" % (precision, recall, f_score))
            if best_f < f_score:
                best_f = f_score
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.logging.info("the model have saved to %s" % (checkpoint_prefix, ))

            # 测试集
            test_labels_pred = test_labels_true = np.array([])
            for _, test_batch_inputs, test_batch_labels in test_dataset.batch_iter(FLAGS.batch_size, 1):
                _, test_predictions = sess.run(
                    [train_op, lstm.predictions],
                    feed_dict={
                        lstm.inputs: test_batch_inputs,
                        lstm.outputs: test_batch_labels,
                        lstm.keep_prob: 1.0
                    })
                test_labels_pred = np.append(test_labels_pred, test_predictions)
                test_labels_true = np.append(test_labels_true, test_batch_labels)

            classify_report = metrics.classification_report(test_labels_true, test_labels_pred)
            precision, recall, f_score = compute_p_r_f(test_labels_true, test_labels_pred)
            tf.logging.info("\nTest: \n%s" % (classify_report,))
            tf.logging.info("\nTest: precision: %3.3f, recall: %3.3f, f_score: %3.3f" % (precision, recall, f_score))
