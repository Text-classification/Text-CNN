import numpy as np
import re
import tensorflow as tf
import itertools
from collections import Counter
import codecs
import random


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 导入数据
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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

# 导入初始词向量
def loadVectors(wordvectorPath):
    vectorFile = codecs.open(wordvectorPath, "r", "utf-8")
    vocab = {}
    embd = []
    line = vectorFile.readline().strip()
    word_dim = int(line.split(" ")[1])

    vocab["***unk***"] = 0
    embd.append([random.uniform(-1.0, 1.0)] * word_dim)

    vocab["***eric***"] = 1
    embd.append([0] * word_dim)

    count = 2
    for line in vectorFile:
        # print(line.strip())
        row = line.strip().split(" ")
        vocab[row[0]] = count
        embd.append(row[1:])
        count += 1
    print("loaded word2vec!!")
    vectorFile.close()
    print("Vocab Size : ", vocab.__len__())
    return vocab, embd

# 重新构造输入
# input_x 为所有的输入样本，列表中的每一个元素是一个样本，中间按空格隔开
#
def reShapeX(vocab, input_x, max_document_length):
    in_x = []
    count = 1
    for x in input_x:
        s = []
        number = 0
        for i in x.split(" "):
            number += 1
            if number > max_document_length:# 设定最大文本长度，超出部分删除
                break
            if vocab.get(i) == None:
                s.append(0)
            else:
                s.append(vocab.get(i))
        if s.__len__() < max_document_length:
            s += [1] * (max_document_length - s.__len__())
        in_x.append(s)
        # print(count)
        count += 1
    return in_x

# 导入中文分类文本数据
def loadChineseInput(inputfile, number_of_class):
    file = codecs.open(inputfile, "r", encoding="utf-8")
    fileLine = file.readlines()
    labelDict = {}
    result = []
    x_text = []
    count = 0
    for line in fileLine:
        line = line.strip()
        labels = line[0: line.index(",")]
        wordList = line[line.index(",") + 1: line.__len__()]
        x_text.append(wordList)
        l = [0] * number_of_class
        for label in labels.split(" "):
            if labelDict.get(label) == None:
                l[count] = 1
                labelDict[label] = count
                count += 1
            else:
                l[labelDict[label]] = 1
        if l.__len__() != number_of_class:
            print("Error")
        result.append(l)
    file.close()
    # for key in labelDict:
    #     print(key, labelDict[key])
    y = np.array(result)
    return [x_text, y]

if __name__ == "__main__":
    # loadVectors("./wiki_english_dim100.vec")
    x_text, y = loadChineseInput("fuzhouFilterDatas.txt")
    print(type(y[1][1]))
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print(max_document_length)

    vocab, embd = loadVectors("wiki_chinese_dim100_seg.vec")

    x = np.array(reShapeX(vocab, x_text, max_document_length))

    print(x)
    # for i in y:
    #     print(i)
    # for x in x_text:
    #     print(x)
