import csv
import re
import string
from collections import Counter
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def preprocess_string(str_arg):
    """"
        Parameters:
        ----------
        str_arg: example string to be preprocessed

        What the function does?
        -----------------------
        Preprocess the string argument - str_arg - such that :
        1. everything apart from letters is excluded
        2. multiple spaces are replaced by single space
        3. str_arg is converted to lower case

        Example:
        --------
        Input :  Menu is absolutely perfect,loved it!
        Output:  ['menu', 'is', 'absolutely', 'perfect', 'loved', 'it']


        Returns:
        ---------
        Preprocessed string

    """

    cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)  # every char except alphabets is replaced
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)  # multiple spaces are replaced by single space
    cleaned_str = cleaned_str.lower()  # converting the cleaned string to lower case

    return cleaned_str  # returning the preprocessed string in tokenized form

# #===========================READ FILE========================================
# tsv_file = open('../Dataset/covid_training.tsv')
# read_tsv = csv.reader(tsv_file, delimiter="\t")
# # from the description, we should label this as V which will be used
# # as features but I label as vocabulary for a better understanding
# values = []
# vocabulary = []
# for i, row in enumerate(read_tsv):
#     #skip the head
#     if(i > 0):
#         #extract each value from column text
#         # and use casefold() to change the values in lower case
#         text_cleaning = preprocess_string(row[1])
#         values = text_cleaning.split()
#         # values = row[1].casefold().split()
#         # print(text_cleaning)
#         # print("================================================================================================================================================================================================")
#         # print(values)
#         for words in values:
#             vocabulary.append(words)
# TODO: add words identification, tokenise the tweets (DONE)

# function to convert a document into list of words
# def count_vectorizer(path):
    # load document
    # tsv_file = open(path)
tsv_file = open('../Dataset/covid_training.tsv')
read_tsv = csv.reader(tsv_file, delimiter="\t")

    # from the description, we should label this as V which will be used
    # as features but I label as vocabulary for a better understanding
values = []
preprocessed_text = []
    # Step: Bag of words
for i, row in enumerate(read_tsv):
        # skip the head (the first row of the document)
    if (i > 0):
            # extract each value from column text and clean text
            # Step: Cleaning text
        text_cleaning = preprocess_string(row[1])
        values = text_cleaning.split()
        for words in values:
            preprocessed_text.append(words)

# print(preprocessed_text)
# Step: Frequency of words
frequency_list = []
frequency_list.append(dict(Counter(preprocessed_text)))
#
# print(frequency_list)

vocabulary = list(set([j for i in preprocessed_text for j in i]))

for text in frequency_list:
    for word in vocabulary:
        if word not in list(text.keys()):
            text[word] = 0

df = pd.DataFrame(frequency_list)
df = df[sorted(list(df.columns))]

vocab = df.columns.to_list()

# print(vocab)

fields = ['text', 'q1_label']
training_set = pd.read_csv('../Dataset/covid_training.tsv',sep='\t', skipinitialspace=True, usecols = fields)
# print(training_set.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_set['text'],
                                                    training_set['q1_label'],
                                                    random_state=1)
print(list(X_test[:10]))



# print(vocabulary)


#================================NB_BOW_OV===================================
# nb_bow_ov =  dict(Counter(vocab))
# print(nb_bow_ov)

#================================NB_BOW_FV===================================
#function to remove elements that its occurrences less than 1
# def removeElements(lst, k):
#     counted = Counter(lst)
#     return [el for el in lst if counted[el] >= k]
#
# new_v = removeElements(vocabulary, 2)
#
# #using dict to remove Counter({}) output
# nb_bow_fv = dict(Counter(new_v))
# # print(nb_bow_fv)

