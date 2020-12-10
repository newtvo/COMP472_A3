import pandas as pd
from Model.preprocessing import *


# split dataset into its own label
def split_text_train(text_in_label):
    data = []
    data_dict ={'0': [], '1': []}
    vocabulary = 0
    spam_occurences = 0
    not_spam_occurences = 0
    sum_data = 0

    for row in text_in_label:
        sum_data += 1

        if row[1] == 1:
            split = row[0].split()
            spam_occurences += 1

            for i in split:
                data.append(i)
                if i not in data_dict['1']:
                    vocabulary += 1
                data_dict['1'].append(i)

        if row[1] == 0:
            split = row[0].split()
            not_spam_occurences += 1

            for i in split:
                data.append(i)
                if i not in data_dict['0']:
                    vocabulary += 1
                data_dict['0'].append(i)

    return data, data_dict, vocabulary, spam_occurences, not_spam_occurences, sum_data

if __name__ == '__main__':

    data = get_words()
    print(data)