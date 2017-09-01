# coding: utf-8
import codecs
import random
import os
from textCNN import data_helpers

def parsed(word2vector, input):
    vocab, embd = data_helpers.loadVectors(word2vector)
    fileList = []
    if os.path.isdir(input):
        for root, dirs, files in os.walk(input):
            for i in files:
                fileList.append(root + "/" + i)
    else:
        fileList.append(input)

    notExist = []
    for i in fileList:
        print(i)
        file = codecs.open(i, "r", encoding="utf-8").readlines()
        for line in file:
            line = line.strip()
            line = data_helpers.clean_str(line)
            for x in line.split(" "):
                if x not in notExist:
                    notExist.append(x)

    print(notExist.__len__())
    Exist = []
    for x in notExist:
        # print(x)
        if x not in vocab:
            Exist.append(x)
    print(Exist.__len__())

    output = codecs.open("newWordvec", "a+", encoding="utf-8")
    for x in Exist:
        # print(x)
        s = ""
        for i in range(100):
            a = random.uniform(-1.0, 1.0)
            s += " " + '%f' % a
        # print(x + s)
        output.writelines(x + s + "\n")
    output.close()


if __name__ == "__main__":
    parsed("wiki_english_dim100.vec",
           "/root/PycharmProjects/TextCNN/textCNN/data/rt-polaritydata")