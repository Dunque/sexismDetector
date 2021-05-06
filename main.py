import fasttext
from src.binaryfuncs import *
from src.multifuncs import *

def main():
    trainingTextBin = import_text("training/EXIST2021_training.tsv")
    
    format_text_bin(trainingTextBin)

    crossValBinEN(5)

    crossValBinES(5)

    testBin()

    trainingTextMulti = import_text("training/EXIST2021_training.tsv")

    format_text_multi(trainingTextMulti)

    crossValMultiEN(5)

    crossValMultiES(5)

    testMulti()

if __name__ =='__main__':
    main()