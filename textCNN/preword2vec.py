# coding: utf-8
import codecs
import os

def parsed(input):
    if os.path.exists("fuzhouFilterData.txt"):
        os.remove("fuzhouFilterData.txt")
    file = codecs.open(input, "r", encoding="utf-8")
    output = codecs.open("fuzhouFilterData.txt",
                         "a+", encoding="utf-8")
    for line in file.readlines():
        line = line.strip()
        i = line[line.index(",") + 1: line.__len__()]
        output.writelines(i + "\n")
    file.close()
    output.close()

if __name__ == "__main__":
    parsed("fuzhouFilterDatas.txt")