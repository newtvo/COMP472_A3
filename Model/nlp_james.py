import re
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords


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

    def count_vectorizer(self, text, train = True, stop_word=None):
        # from the description, we should label this as V which will be used
        # as features but I label as vocabulary for a better understanding
        documents = text
        values = []
        preprocessed_text = []
        if stop_word == None:
          stop_word = set(stopwords.words('english'))
        # Step: Bag of words
        for i in documents:
            # skip the head (the first row of the document)
            # if (i > 0):
                # extract each value from column text and clean text
                # Step: Cleaning text
                text_cleaning = self.preprocess_string(i)
                values = text_cleaning.split()
                for words in values:
                    if words not in stop_word:
                        preprocessed_text.append(words)

        # Step: Frequency of words
        frequency_list = []
        frequency_list.append(dict(Counter(preprocessed_text)))

        # print(frequency_list)

        vocabulary = list(set([j for i in preprocessed_text for j in i]))

        for text in frequency_list:
            for word in vocabulary:
                if word not in list(text.keys()):
                    text[word] = 0

        df = pd.DataFrame(frequency_list)
        df = df[sorted(list(df.columns))]

        vocab = df.columns.to_list()
        return df

fields = ['text', 'q1_label']
training_set = pd.read_csv('../Dataset/covid_training.tsv',sep='\t', skipinitialspace=True, usecols = fields)
print(training_set.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_set['text'],
                                                    training_set['q1_label'],
                                                    random_state=1)
nlp = NLP()
count_vector = nlp.count_vectorizer(list(X_test[:100]))
print(count_vector)
# print(list(X_test[:10]))



