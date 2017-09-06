#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# 参数
# ==================================================

# 输入数据参数
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")
tf.flags.DEFINE_string("chinese_data_file", "./xaa", "Data source for the chinese classification")
tf.flags.DEFINE_integer("max_document_length", 698, "Max length of each document")
tf.flags.DEFINE_integer("number_of_class", 141, "Number of class")

# Checkpoint
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1504676385/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc 参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 初始训练的词向量
tf.flags.DEFINE_string("pre_trained_wordvector", "./fuzhou_dim100.vec", "Pre-trained word vector")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 导入验证数据集
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.loadChineseInput(FLAGS.chinese_data_file, FLAGS.number_of_class)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["霞拔 中心小学 2017 年 月 中旬 评比 市 县级 先进 教师 不合理 现象 校长 想评 评先 文件 评选 陈静虹 老师 毕业 一年 班主任 评为 县 先进 班主任 先进 评给 表现 先进 教师 打击 教师 积极性 呼吁 部门 明察秋毫 纠正 评先 评先 不合理"]
    y_test = [1]
    y_test += [0] * 140

# 导入词向量
vocab, embd = data_helpers.loadVectors(FLAGS.pre_trained_wordvector)

# 将输入数据与词向量进行转换
max_document_length = FLAGS.max_document_length
x_test = np.array(data_helpers.reShapeX(vocab, x_raw, max_document_length))

print("\nEvaluating...\n")

# 开始验证
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 导入训练好的模型与变量
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        # 通过名字导入变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        W = graph.get_operation_by_name("embedding/WW").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # 验证的Tensor
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # 生成批量数据集
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        # 准确率集合
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, W : embd, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# 输出准确率
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# 保存
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)