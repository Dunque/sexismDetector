import fasttext
import io
import csv
import re
from nltk.corpus import stopwords
import pandas as pd

def import_text(filename):
    for line in csv.reader(open(filename, encoding="utf-8"), delimiter="\t"):
        if line:
            yield line

def remove_clutterEN(text):
    # keep only words
    remove_links = re.sub(r"(?:\@|https?\://)\S+", "", text)
    letters_only_text = re.sub("[^a-zA-Z]", " ", remove_links)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # join the cleaned words in a list
    return " ".join(meaningful_words)

def remove_clutterES(text):
    # keep only words
    remove_links = re.sub(r"(?:\@|https?\://)\S+", "", text)
    letters_only_text = re.sub("[^abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZáéíóúüç]", " ", remove_links)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # join the cleaned words in a list
    return " ".join(words)

def format_text(text):
    trainES = open("training/binaryES.train", 'w', encoding="utf-8")
    trainEN = open("training/binaryEN.train", 'w', encoding="utf-8")

    lines = list(text)
    lines = lines[1:]

    es = []
    en = []

    for line in lines:
        if line[3] == "en":
            en.append(line)
        else:
            es.append(line)

    for line in en:
        trainEN.write(remove_clutterEN(line[4]) + ' __label__' + line[5]+ "\n") 
    for line in es:
        trainES.write(remove_clutterES(line[4]) + ' __label__' + line[5]+ "\n")

def crossValEN(k):
    print("Strating English cross validation\n")
    trainSet = list(import_text("training/binaryES.train"))

    size = len(trainSet)

    ksize = int(size/k)

    acc_score = []

    a = 0
    b = ksize

    for x in range(1,6):

        trainEN = open("training/binaryFold.train", 'w', encoding="utf-8")
        validEN = open("training/binaryFold.valid", 'w', encoding="utf-8")

        enTrain = trainSet[0:a] + trainSet[b:size]
        enVal = trainSet[a:b]

        for line in enTrain:
            trainEN.write(str(line) + '\n')

        for line in enVal:
            validEN.write(str(line) + '\n')

        if (len(acc_score) == 0):
            model = fasttext.load_model("binary_classEN.bin")

        model = fasttext.train_supervised(input="training/binaryFold.train", epoch=25, lr=1.0, wordNgrams=3,
        bucket=200000, dim=50, loss='hs')

        metrics = model.test("training/binaryFold.valid")
        
        if all(i < metrics[1] for i in acc_score):
            print("Saved model!")
            print(metrics[1])
            model.save_model("binary_classEN.bin")

        acc_score.append(metrics[1])

        a += ksize
        b += ksize
        trainEN.close()
        validEN.close()

    avg_acc_score = sum(acc_score)/k
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    print('\n')

def crossValES(k):
    print("Strating English cross validation\n")
    trainSet = list(import_text("training/binaryES.train"))

    size = len(trainSet)

    ksize = int(size/k)

    acc_score = []

    a = 0
    b = ksize

    for x in range(1,6):

        trainEN = open("training/binaryFold.train", 'w', encoding="utf-8")
        validEN = open("training/binaryFold.valid", 'w', encoding="utf-8")

        enTrain = trainSet[0:a] + trainSet[b:size]
        enVal = trainSet[a:b]

        for line in enTrain:
            trainEN.write(str(line) + '\n')

        for line in enVal:
            validEN.write(str(line) + '\n')

        if (len(acc_score) == 0):
            model = fasttext.load_model("binary_classEN.bin")

        model = fasttext.train_supervised(input="training/binaryFold.train", epoch=25, lr=1.0, wordNgrams=3,
        bucket=200000, dim=50, loss='hs')

        metrics = model.test("training/binaryFold.valid")
        
        if all(i < metrics[1] for i in acc_score):
            print("Saved model!")
            print(metrics[1])
            model.save_model("binary_classEN.bin")

        acc_score.append(metrics[1])

        a += ksize
        b += ksize
        trainEN.close()
        validEN.close()

    avg_acc_score = sum(acc_score)/k
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    print('\n')

def test(text):
    output = open("test/binaryTest.tsv", 'w', encoding="utf-8")

    lines = list(text)
    lines = lines[1:]

    modelEN = fasttext.load_model("binary_classEN.bin")
    modelES = fasttext.load_model("binary_classES.bin")

    for line in lines:
        if line[3] == "en":
            labels = modelEN.predict(remove_clutterEN(line[4]), k=1)
            label = labels[0][0]
            label = label.replace("__label__", "")
            label = label.replace("']", "")
            output.write("EXIST2021 " + line[1] + " " + label + "\n")
        else:
            label = modelES.predict(remove_clutterES(line[4]), k=1)
            label = labels[0][0]
            label = label.replace("__label__", "")
            label = label.replace("']", "")
            output.write("EXIST2021 " + line[1] + " " + label + "\n")

def main():
    trainingText = import_text("training/EXIST2021_training.tsv")
    testText = import_text("test/EXIST2021_test.tsv")

    format_text(trainingText)

    crossValEN(5)

    crossValES(5)

    test(testText)


if __name__ =='__main__':
    main()

