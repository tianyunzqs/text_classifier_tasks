#! /usr/bin/env python

import os
import sys
import datetime
import pickle
import json
import time
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn import metrics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_cnn import TextCNN


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .98, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("config_file", "./config_file", "config file for model")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open("./data/train_data.pickle", 'rb') as f:
    train_data = pickle.loads(f.read())
with open("./data/test_data.pickle", 'rb') as f:
    test_data = pickle.loads(f.read())

classify = {}
tags = {}
for i, cls in enumerate(train_data.keys()):
    tmp = [0] * len(train_data.keys())
    tmp[i] = 1
    classify[cls] = tmp
    tags[i] = cls

x_train, y_train = [], []
for cls, data in train_data.items():
    x_train.extend(data)
    y_train.extend([classify[cls]] * len(data))
x_test, y_test = [], []
for cls, data in test_data.items():
    # data = data[:int(0.5 * len(data))]
    x_test.extend(data)
    y_test.extend([classify[cls]] * len(data))

t1 = time.time()
max_document_length = max([len(x.split(" ")) for x in x_train])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
y_train = np.array(y_train)
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# config for the model
def config_model(x_train, y_train):
    config = OrderedDict()

    config["allow_soft_placement"] = FLAGS.allow_soft_placement
    config["log_device_placement"] = FLAGS.log_device_placement

    config["sequence_length"] = x_train.shape[1]
    config["num_classes"] = y_train.shape[1]

    config["embedding_dim"] = FLAGS.embedding_dim
    config["filter_sizes"] = FLAGS.filter_sizes
    config["num_filters"] = FLAGS.num_filters
    config["dropout_keep_prob"] = FLAGS.dropout_keep_prob
    config["l2_reg_lambda"] = FLAGS.l2_reg_lambda
    config["batch_size"] = FLAGS.batch_size

    config["tags"] = tags
    return config


config = config_model(x_train, y_train)
with open(FLAGS.config_file, "w", encoding="utf8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # # Keep track of gradient values and sparsity (optional)
        # grad_summaries = []
        # for g, v in grads_and_vars:
        #     if g is not None:
        #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        #         grad_summaries.append(grad_hist_summary)
        #         grad_summaries.append(sparsity_summary)
        # grad_summaries_merged = tf.summary.merge(grad_summaries)
        #
        # # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        # print("Writing to {}\n".format(out_dir))
        #
        # # Summaries for loss and accuracy
        # loss_summary = tf.summary.scalar("loss", cnn.loss)
        # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        #
        # # Train Summaries
        # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        # train_summary_dir = os.path.join(out_dir, "summaries", "train")
        # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        #
        # # Dev summaries
        # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "text_cnn_model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save("./models/vocab.model")

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        fout = open("log.txt", "w", encoding="utf-8")

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            fout.write("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
            fout.flush()
            # train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            accuracy = sess.run(cnn.accuracy, feed_dict)
            return accuracy

        def test_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            predictions, accuracy = sess.run([cnn.predictions, cnn.accuracy], feed_dict)
            return predictions, accuracy

        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]

        # Generate batches
        batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...

        best_acc = 0.0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                fout.write("\nEvaluation:")
                fout.flush()
                y_true, y_pred, y_acc = [], [], 0
                for x, y in zip(x_test, y_test):
                    data = np.array(list(vocab_processor.transform([x])))
                    pred, acc = test_step(data, np.array([y]))
                    y_acc += acc
                    y_true.append(tags[y.index(max(y))])
                    y_pred.append(tags[pred[0]])
                classify_report = metrics.classification_report(y_true, y_pred)
                y_acc = y_acc / len(y_true)
                print(classify_report)
                print(y_acc)
                fout.write(str(classify_report))
                fout.write(str(y_acc))
                fout.flush()

                if y_acc > best_acc:
                    best_acc = y_acc
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoints to {}\n".format(path))
                    fout.write("Saved model checkpoints to {}\n".format(path))
                    fout.flush()

t2 = time.time()
print('train model over. it took {0}ms'.format((t2 - t1)))
