import fasttext
import io
import csv
import re
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
#Implementing cross validation
 
k = 5
kf = KFold(n_splits=k, random_state=None)
model = LogisticRegression(solver= 'liblinear')
 
acc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))

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

def format_text(text, tvratio):
    trainES = open("training/binaryES.train", 'w', encoding="utf-8")
    trainEN = open("training/binaryEN.train", 'w', encoding="utf-8")
    validES = open("training/binaryES.valid", 'w', encoding="utf-8")
    validEN = open("training/binaryEN.valid", 'w', encoding="utf-8")

    lines = list(text)
    lines = lines[1:]

    es = []
    en = []
    n = 0

    for line in lines:
        n += 1
        if line[3] == "en":
            en.append(line)
        else:
            es.append(line)

    vn = round(tvratio * n)

    esTrain = es[:len(es)-vn]
    esVal = es[-vn:]

    enTrain = en[:len(en)-vn]
    enVal = en[-vn:]

    unwanted = ["*","+","-","/","!","?","º",
    "ª","|","·","~",",",".",":",";","_","0",
    "1","2","3","4","5","6","7","8","9"]

    for line in enTrain:
        #auxline = line[4].lower()
        #translation = auxline.maketrans({i:"" for i in unwanted})
        #trainEN.write(auxline.translate(translation) + ' __label__' + line[5]+ "\n")
        trainEN.write(remove_clutterEN(line[4]) + ' __label__' + line[5]+ "\n") 
    for line in esTrain:
        #auxline = line[4].lower()
        #translation = auxline.maketrans({i:"" for i in unwanted})
        #trainES.write(auxline.translate(translation) + ' __label__' + line[5]+ "\n")
        trainES.write(remove_clutterES(line[4]) + ' __label__' + line[5]+ "\n")
        #trainES.write(line[4] + ' __label__' + line[5]+ "\n") 
    for line in enVal:
        #auxline = line[4].lower()
        #translation = auxline.maketrans({i:"" for i in unwanted})
        #validEN.write(auxline.translate(translation) + ' __label__' + line[5]+ "\n")
        validEN.write(remove_clutterEN(line[4]) + ' __label__' + line[5]+ "\n")
    for line in esVal:
        #auxline = line[4].lower()
        #translation = auxline.maketrans({i:"" for i in unwanted})
        #validES.write(auxline.translate(translation) + ' __label__' + line[5]+ "\n")
        #validES.write(line[4] + ' __label__' + line[5]+ "\n") 
        validES.write(remove_clutterES(line[4]) + ' __label__' + line[5]+ "\n")

def main():
    text = import_text("training/EXIST2021_training.tsv")
    #tvratio means train to validate ratio %, 0.2 amounts to 20% validate 80% train
    format_text(text, 0.15)

    modelEN = fasttext.train_supervised(input="training/binaryEN.train", epoch=25, lr=1.0, wordNgrams=3,
        bucket=200000, dim=50, loss='hs')

    modelES = fasttext.train_supervised(input="training/binaryES.train", epoch=25, lr=1.0, wordNgrams=3,
        bucket=200000, dim=50, loss='hs')

    modelEN.save_model("binary_classEN.bin")
    modelES.save_model("binary_classES.bin")

    def print_results(N, p, r):
        print("N \t" + str(N))
        print("Precision @{}\t{:.3f}".format(1, p))
        print("Recall @{}\t{:.3f}".format(1, r))

    print_results(*modelEN.test("training/binaryEN.valid"))
    print_results(*modelES.test("training/binaryES.valid"))

    #testText = import_text("training/EXIST2021_training.tsv")

if __name__ =='__main__':
    main()

