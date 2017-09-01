import numpy as np
import re
import tensorflow as tf
import itertools
from collections import Counter
import codecs


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

def loadVectors(wordvectorPath):
    vectorFile = codecs.open(wordvectorPath, "r", "utf-8")
    vocab = []
    embd = []
    line = vectorFile.readline().strip()
    word_dim = int(line.split(" ")[1])

    vocab.append("unk")
    embd.append([0.0] * word_dim)

    for line in vectorFile:
        # print(line.strip())
        row = line.strip().split(" ")
        vocab.append(row[0])
        embd.append(row[1:])
    print("loaded word2vec!!")
    vectorFile.close()
    print("Vocab Size : ", vocab.__len__())
    return vocab, embd

def reShapeX(vocab : list, input_x, max_document_length):
    in_x = []
    for x in input_x:
        s = []
        for i in x.split(" "):
            s.append(vocab.index(i))
        if s.__len__() < max_document_length:
            for c in range(max_document_length - s.__len__()):
                s.append(0)
        # print(s.__len__(), max_document_length)
        in_x.append(s)
    return in_x

if __name__ == "__main__":
    loadVectors("./wiki_english_dim100.vec")
