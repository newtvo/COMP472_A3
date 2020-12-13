import copy
import csv
import re
from collections import Counter


def get_test_file():
    tsv_file = open('./Dataset/covid_test_public.tsv', encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    return read_tsv

def get_text_file():
    tsv_file = open('./Dataset/covid_training.tsv', encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    return read_tsv

def get_words():
    list_of_words = []
    for row in get_text_file():
        for word in clean_text(row[1]):
            list_of_words.append(word)
    return list_of_words


def get_original_vocabulary():
    original_vocabulary = set(get_words())
    return original_vocabulary


def get_filtered_vocalbulary():
    non_filtered_vocabulary = dict(Counter(get_words()))
    filtered_vocabulary = dict(filter(lambda elem: elem[1] >= 2, non_filtered_vocabulary.items()))
    return set(filtered_vocabulary.keys())


def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z\s]+', ' ', text, flags=re.IGNORECASE)  # every char except alphabets is replaced
    text = re.sub('(\s+)', ' ', text)  # multiple spaces are replaced by single space
    return text.split(" ")


ORIGINAL = get_original_vocabulary()
FILTERED = get_filtered_vocalbulary()

def vectorize_text(text, filtered=False):
    text = clean_text(text)
    if filtered:
        vector = copy.deepcopy(dict.fromkeys(FILTERED, 0))
    else:
        vector = copy.deepcopy(dict.fromkeys(ORIGINAL, 0))
    for word in text:
        vector[word] += 1
    return vector

def get_features():
    X = []
    texts = get_text_file()
    next(texts, None)
    for row in texts:
        X.append(vectorize_text(row[1]))
    return X
def get_values():
    y= []
    texts = get_text_file()
    next(texts, None)
    for row in texts:
        if row[2] == "yes":
            y.append(1)
        else:
            y.append(0)
    return y
# if __name__ == '__main__':
#     # print(len(ORIGINAL))
#     # print(get_features()[0])
#     print(len(get_features()))
#     print(len(get_values()))