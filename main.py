import fasttext
import io
import csv

class Data:
    def __init__(self,test_case,id,source,language,text,task1,task2):
        self.test_case = test_case
        self.id = id
        self.source = source
        self.language = language
        self.text = text
        self.task1 = task1
        self.task2 = task2

def import_text(filename):
    for line in csv.reader(open(filename, encoding="utf-8"), delimiter="\t"):
        if line:
            yield line

def format_text(text, tvratio):
    train = open("training/binary.train", 'w', encoding="utf-8")
    valid = open("training/binary.valid", 'w', encoding="utf-8")

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

    vn = round(tvratio * n/2)

    esTrain = es[:len(es)-vn]
    esVal = es[-vn:]

    enTrain = en[:len(en)-vn]
    enVal = en[-vn:]

    unwanted = ["*","+","-","/","!","?","º",
    "ª","|","·","~",",",".",":",";"]

    for line in enTrain:
        auxline = line[4] + ' __label__' + line[5]+ "\n"
        translation = auxline.maketrans({i:"" for i in unwanted})
        #train.write(auxline.translate(translation))
        train.write(line[4] + ' __label__' + line[5]+ "\n") 
    for line in esTrain:
        train.write(line[4] + ' __label__' + line[5]+ "\n") 
    for line in enVal:
        valid.write(line[4] + ' __label__' + line[5]+ "\n")
    for line in esVal:
        valid.write(line[4] + ' __label__' + line[5]+ "\n")

def main():
    text = import_text("training/EXIST2021_training.tsv")
    #tvratio means train to validate ratio %, 0.2 amounts to 20% validate 80% train
    format_text(text, 0.2)

    model = fasttext.train_supervised(input="training/binary.train", epoch=25, lr=1.0, wordNgrams=3,
        bucket=200000, dim=50, loss='hs')

    model.save_model("binary_class.bin")
    print(model.test("training/binary.valid"))

if __name__ =='__main__':
    main()