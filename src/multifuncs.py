import fasttext
import io
import csv
import re
from nltk.corpus import stopwords

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

    # remove stopwords
    stopword_set = set(stopwords.words("spanish"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # join the cleaned words in a list
    return " ".join(meaningful_words)

def format_text_multi(text):
    trainES = open("training/multiES.train", 'w', encoding="utf-8")
    trainEN = open("training/multiEN.train", 'w', encoding="utf-8")

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
        trainEN.write(remove_clutterEN(line[4]) + ' __label__' + line[6]+ "\n") 
    for line in es:
        trainES.write(remove_clutterES(line[4]) + ' __label__' + line[6]+ "\n")

def crossValMultiEN(k):
    print("Strating English cross validation\n")
    trainSet = list(import_text("training/multiES.train"))

    size = len(trainSet)

    ksize = int(size/k)

    acc_score = []

    a = 0
    b = ksize

    for x in range(1,6):

        trainEN = open("training/multiFold.train", 'w', encoding="utf-8")
        validEN = open("training/multiFold.valid", 'w', encoding="utf-8")

        enTrain = trainSet[0:a] + trainSet[b:size]
        enVal = trainSet[a:b]

        for line in enTrain:
            trainEN.write(str(line) + '\n')

        for line in enVal:
            validEN.write(str(line) + '\n')

        # if (len(acc_score) == 0):
        #     try: 
        #         model = fasttext.load_model("classifiers/multi_classEN.bin")
        #     except:
        #         print("Couldn't load model")
        
        model = fasttext.train_supervised(input="training/multiFold.train", epoch=25, lr=0.5, wordNgrams=3,
        bucket=200000, dim=50, loss='ova')

        metrics = model.test("training/multiFold.valid")
        
        score = (2*metrics[1]*metrics[2])/(metrics[1]+metrics[2])
        
        if all(i < score for i in acc_score):
            print("Saved model!")
            print(score)
            model.save_model("classifiers/multi_classEN.bin")

        acc_score.append(score)

        a += ksize
        b += ksize
        trainEN.close()
        validEN.close()

    avg_acc_score = sum(acc_score)/k
    print('F1 of each fold - {}'.format(acc_score))
    print('Avg F1 : {}'.format(avg_acc_score))
    print('\n')

def crossValMultiES(k):
    print("Strating Spanish cross validation\n")
    trainSet = list(import_text("training/multiES.train"))

    size = len(trainSet)

    ksize = int(size/k)

    acc_score = []

    a = 0
    b = ksize

    for x in range(1,6):

        trainEN = open("training/multiFold.train", 'w', encoding="utf-8")
        validEN = open("training/multiFold.valid", 'w', encoding="utf-8")

        enTrain = trainSet[0:a] + trainSet[b:size]
        enVal = trainSet[a:b]

        for line in enTrain:
            trainEN.write(str(line) + '\n')

        for line in enVal:
            validEN.write(str(line) + '\n')

        # if (len(acc_score) == 0):
        #     try: 
        #         model = fasttext.load_model("classifiers/multi_classES.bin")
        #     except:
        #         print("Couldn't load model")
        
        model = fasttext.train_supervised(input="training/multiFold.train", epoch=25, lr=0.5, wordNgrams=3,
        bucket=200000, dim=50, loss='ova')

        metrics = model.test("training/multiFold.valid")

        score = (2*metrics[1]*metrics[2])/(metrics[1]+metrics[2])
        
        if all(i < score for i in acc_score):
            print("Saved model!")
            print(score)
            model.save_model("classifiers/multi_classES.bin")

        acc_score.append(score)

        a += ksize
        b += ksize
        trainEN.close()
        validEN.close()

    avg_acc_score = sum(acc_score)/k
    print('F1 of each fold - {}'.format(acc_score))
    print('Avg F1 : {}'.format(avg_acc_score))
    print('\n')

def testMulti():
    print("Creating output")
    testTextMulti = import_text("test/EXIST2021_test.tsv")

    output = open("test/multiTest.tsv", 'w', encoding="utf-8")

    modelEN = fasttext.load_model("classifiers/multi_classEN.bin")
    modelES = fasttext.load_model("classifiers/multi_classES.bin")

    lines2 = list(testTextMulti)
    lines2 = lines2[1:]

    for line in lines2:
        if line[3] == "en":
            labels = modelEN.predict(remove_clutterEN(line[4]), k=1)
            label = labels[0][0]
            label = label.replace("__label__", "")
            label = label.replace("']", "")
            output.write("EXIST2021\t" + line[1] + "\t" + label + "\n")
        else:
            label = modelES.predict(remove_clutterES(line[4]), k=1)
            label = labels[0][0]
            label = label.replace("__label__", "")
            label = label.replace("']", "")
            output.write("EXIST2021\t" + line[1] + "\t" + label + "\n")
    print("Finished creating output")