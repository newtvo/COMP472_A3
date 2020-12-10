"""
Links: - https://towardsdatascience.com/different-techniques-to-represent-words-as-vectors-word-embeddings-3e4b9ab7ceb4
       - https://github.com/JKnighten/text-naive-bayes/blob/master/naivebayes/models/dictionary.py
       - https://github.com/JKnighten/text-naive-bayes
       - https://github.com/aishajv/Unfolding-Naive-Bayes-from-Scratch/blob/master/%23%20Unfolding%20Na%C3%AFve%20Bayes%20from%20Scratch!%20Take-2%20%F0%9F%8E%AC.ipynb
"""
import math

import pandas as pd
import numpy as np
from collections import defaultdict
from Model.preprocessing import *

# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:
        def __init__(self, features, values, vocabulary):
            self.features = features
            self.values = values
            self.vocabulary = vocabulary
            self.conditional_probality_given_yes, self.conditional_probality_given_no = self.train()
            self.yes_probabiliy = values.count(1) / len(values)
            self.smoothing = 0.01

        def train(self):
            conditional_probability_given_yes = copy.deepcopy(dict.fromkeys(self.vocabulary, 0))
            conditional_probability_given_no = copy.deepcopy(dict.fromkeys(self.vocabulary, 0))

            for word in self.vocabulary:
                for i in range(len(self.features)):
                    if self.values[i] == 1:
                        conditional_probability_given_yes[word] += self.features[i][word]
                    else:
                        conditional_probability_given_no[word] += self.features[i][word]
            return conditional_probability_given_yes, conditional_probability_given_no

        def predict(self, text):
            yes_score = math.log10(self.yes_probabiliy)
            no_score = math.log10(1 - self.yes_probabiliy)
            for word in text:
                if word in self.vocabulary:
                    yes_score += math.log10((self.conditional_probality_given_yes[word] + self.smoothing) / (
                            self.get_total_vocab_of_class(1) + len(self.vocabulary)))
                    no_score += math.log10((self.conditional_probality_given_no[word] + self.smoothing) / (
                            self.get_total_vocab_of_class(0) + len(self.vocabulary)))
            return 1 if yes_score > no_score else 0

        def get_total_vocab_of_class(self, target):
            count = 0
            for i in range(len(self.features)):
                if self.values[i] == target:
                    for word in self.features[i]:
                        if self.features[i][word] != 0:
                            count += self.features[i][word]
            return count

        # def classification_report(self, text):
            # correct = 0
            # total number of classified message
            # number of correct classified message
            # accuracy = correct / total







if __name__ == '__main__':
    #     X = get_features()
    #     y = get_values()
    #     vocalbulary = get_original_vocabulary()
    #     original_model = Model(X, y, vocalbulary)
    #     print(original_model.predict(clean_text("maybe if i develop feelings for covid-19 it will leave")))

    df = pd.read_csv('../Dataset/covid_training.tsv', sep='\t', header=None)
    # x_test = dg.iloc[:, 1]
    # y_test = dg.iloc[:, 2]

    # print(x_test, y_test)
    print(df.shape)
    df.head()