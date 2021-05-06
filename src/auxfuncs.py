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

    # join the cleaned words in a list
    return " ".join(words)