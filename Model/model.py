from Model.preprocessing import *

import math
class MNB_Model:
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
                            self.get_total_vocab_of_class(1) + (self.smoothing *len(self.vocabulary))))
                no_score += math.log10((self.conditional_probality_given_no[word] + self.smoothing) / (
                            self.get_total_vocab_of_class(0) + (self.smoothing *len(self.vocabulary))))
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