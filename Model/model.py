from Model.preprocessing import *

import math


class MNB_Model:
    def __init__(self, features, values, vocabulary):
        self.features = features
        self.values = values
        self.vocabulary = vocabulary
        self.frequency_in_yes_class, self.frequency_in_no_class = self.train()
        self.yes_probability = values.count(1) / len(values)
        self.smoothing = 0.01

    def train(self):
        frequency_in_yes_class = copy.deepcopy(dict.fromkeys(self.vocabulary, 0))
        frequency_in_no_class = copy.deepcopy(dict.fromkeys(self.vocabulary, 0))
        for word in self.vocabulary:
            for i in range(len(self.features)):
                if self.values[i] == 1:
                    frequency_in_yes_class[word] += self.features[i][word]
                else:
                    frequency_in_no_class[word] += self.features[i][word]
        return frequency_in_yes_class, frequency_in_no_class

    def predict(self, text):
        yes_score = math.log10(self.yes_probability)
        no_score = math.log10(1 - self.yes_probability)
        for word in text:
            if word in self.vocabulary:
                yes_score += math.log10((self.frequency_in_yes_class[word] + self.smoothing) / (
                        self.get_total_vocab_of_class(1) + (self.smoothing * len(self.vocabulary))))
                no_score += math.log10((self.frequency_in_no_class[word] + self.smoothing) / (
                        self.get_total_vocab_of_class(0) + (self.smoothing * len(self.vocabulary))))
        result = 1 if yes_score > no_score else 0
        return result, self.convert_to_sciencetific_notation(yes_score), self.convert_to_sciencetific_notation(no_score)

    def get_total_vocab_of_class(self, target):
        count = 0
        for i in range(len(self.features)):
            if self.values[i] == target:
                for word in self.features[i]:
                    if self.features[i][word] != 0:
                        count += self.features[i][word]
        return count

    def convert_to_sciencetific_notation(self, number):
        return "{:.2E}".format(number)
# if __name__ == '__main__':
#     X = get_features()
#     y = get_values()
#     vocabulary = get_original_vocabulary()
#     original_model = MNB_Model(X, y, vocabulary)
#     print(original_model.predict(clean_text("maybe if i develop feelings for covid-19 it will leave")))
