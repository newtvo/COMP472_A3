import copy
import csv
import re
from collections import Counter

def get_text_file():
    tsv_file = open('../Dataset/covid_training.tsv', encoding="utf8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    return read_tsv

def get_words():
    list_of_words = []
    for row in get_text_file():
        if row[1] == 1:
            split = row[0].s
        # for word in row[1]:
        #     list_of_words.append(word)
    return list_of_words