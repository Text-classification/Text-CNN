#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import text_cnn
from tensorflow.contrib import learn

# 参数
# ==================================================

# 输入数据参数
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("chinese_data_file", "../fuzhouFilterDatas.txt", "Data source for the chinese classification")
tf.flags.DEFINE_integer("max_document_length", 698, "Max length of each document")
tf.flags.DEFINE_integer("number_of_class", 141, "Number of class")
# tf.flags.DEFINE_string("chinese_data_file", "./xaa", "Data source for the chinese classification")

# 模型的超参数
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate (default: 1e-4)")
tf.flags.DEFINE_float("learning_rate_decay", 0.99, "learning rate decay")

# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 训练好的词向量
# tf.flags.DEFINE_string("pre_trained_wordvector", "./wiki_chinese_dim100_seg.vec", "Pre-trained word vector")
tf.flags.DEFINE_string("pre_trained_wordvector", "../fuzhou_dim100.vec", "Pre-trained word vector")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# 数据准备阶段
# ==================================================

# 导入数据
print("Loading data...")
# x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y = data_helpers.loadChineseInput(FLAGS.chinese_data_file, FLAGS.number_of_class)

# 构建词向量与词典
vocab, embd = data_helpers.loadVectors(FLAGS.pre_trained_wordvector)
# max_document_length = max([len(x.split(" ")) for x in x_text])
# print(max_document_length)
max_document_length = FLAGS.max_document_length
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))
x = np.array(data_helpers.reShapeX(vocab, x_text, max_document_length))

# 随机shuffle数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
x_train = x_shuffled
y_train = y_shuffled

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_dev = x_shuffled[dev_sample_index:]
y_dev = y_shuffled[dev_sample_index:]


# 训练
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        # 队列形式运行
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        cnn = text_cnn.TextCNN(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size = len(vocab_processor.vocabulary_),
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            vocab = vocab,
            embd = embd,
            l2_reg_lambda = FLAGS.l2_reg_lambda)

        # 定义训练阶段
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,    # 基础学习率
            global_step,            # 当前迭代的轮数
            x_train.shape[1] / FLAGS.batch_size,    # 过完所有的训练数据需要的迭代次数
            FLAGS.learning_rate_decay   # 学习率衰减速度
        )
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # 收集loss和acc
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # 训练阶段数据收集
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint输出路径
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 输出化所有的变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            训练的每一步
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.W: embd,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step,
                 train_summary_op,
                 cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.W : embd,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op,
                 cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # 获取批量输入数据
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # 训练每一批的数据
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, #writer=dev_summary_writer
                         )
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        coord.request_stop()
        coord.join(threads)