import re
import string
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Model.multi_naive_bayes_james import MultinomialNaiveBayes


class NLP():
    """
    - A NLP class to perform count_vectorizer
    - This class will take a text from any documents ( here is text from the dataset) and clean it before train the data
    """
    def __init__(self):
        self.vocab = None

    def preprocess_string(self, str_arg):
        cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)  # every char except alphabets is replaced
        cleaned_str = re.sub('(\s+)', ' ', cleaned_str)  # multiple spaces are replaced by single space
        cleaned_str = cleaned_str.lower()  # converting the cleaned string to lower case

        return cleaned_str  # returning the preprocessed string in tokenized form

    def tokenize_word(self, text, train = True, stop_word=None):
        # from the description, we should label this as V which will be used
        # as features but I label as vocabulary for a better understanding
        documents = text
        values = []
        preprocessed_text = []
        docs = []
        if stop_word == None:
          stop_word = set(stopwords.words('english'))
        # Step: Bag of words
        for i in documents:
            # skip the head (the first row of the document)
            # if (i > 0):
                # extract each value from column text and clean text
                # Step: Cleaning text
                text_cleaning = self.preprocess_string(i)

                docs.append(text_cleaning)
                values = text_cleaning.split()
                for words in values:
                    if words not in stop_word:
                        preprocessed_text.append(words)

        print(docs)

        # Step: Frequency of words
        # frequency_list = []
        # frequency_list.append(dict(Counter(preprocessed_text)))

        # print(frequency_list)

        vocab = dict(Counter(preprocessed_text))

        # vocabulary = list(set([j for i in preprocessed_text for j in i]))

        vectorizer = CountVectorizer()
        vectorizer.fit(docs)
        x = vectorizer.transform(docs)
        print(vectorizer.vocabulary_)
        print(x.toarray())
        vector = x.toarray()
        print(x)
        return vector

        # print(vocab)
        #
        # features = []
        # for key in vocab:
        #     features.append(key)
        #
        # # print(features)
        #
        # return features


def text_cleaning(str):
    remove_punctuation = [char for char in str if char not in string.punctuation ]

    remove_punctuation = ''.join(remove_punctuation)
    return [word.lower() for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]

document = pd.read_csv('../Dataset/covid_training.tsv', sep = "\t")

X_train, X_test, Y_train, Y_test = train_test_split(document['text'],
                                                    document['q1_label'],
                                                    random_state=None)
# print(document.iloc[:,1].apply(text_cleaning))

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_cleaning).fit(document['text'])

# print(bow_transformer.vocabulary_)
# text_bow = bow_transformer.transform(document['text'])
# text_bow_toarray = text_bow.toarray()
# print(text_bow_toarray)


#

X_train_dataset_bow_transformer = CountVectorizer(analyzer=text_cleaning).fit(X_train)
X_train_dataset = X_train_dataset_bow_transformer.transform(X_train).toarray()
X_test_dataset_bow_transformer = CountVectorizer(analyzer=text_cleaning).fit(X_test)
X_test_dataset = X_train_dataset_bow_transformer.transform(X_test).toarray()


clf2 = MultinomialNaiveBayes()
clf2.fit(X_train_dataset,Y_train)
Y_test_pred = clf2.predict(X_test_dataset)

print("Naive Bayes Classification accuracy: ", classification_report(Y_test, Y_test_pred))

