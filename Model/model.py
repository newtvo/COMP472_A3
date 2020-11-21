import csv
from collections import Counter

#===========================READ FILE========================================
tsv_file = open('../Dataset/covid_training.tsv')
read_tsv = csv.reader(tsv_file, delimiter="\t")
# from the description, we should label this as V which will be used
# as features but I label as vocabulary for a better understanding
values = []
vocabulary = []
for i, row in enumerate(read_tsv):
    #skip the head
    if(i > 0):
        #extract each value from column text
        # and use casefold() to change the values in lower case
        values = row[1].casefold().split()
        for words in values:
            vocabulary.append(words)
# TODO: add words identification, tokenise the tweets (DONE)
#================================NB_BOW_OV===================================
nb_bow_ov =  dict(Counter(vocabulary))
print(nb_bow_ov)

#================================NB_BOW_FV===================================
#function to remove elements that its occurrences less than 1
def removeElements(lst, k):
    counted = Counter(lst)
    return [el for el in lst if counted[el] >= k]

new_v = removeElements(vocabulary, 2)

#using dict to remove Counter({}) output
nb_bow_fv = dict(Counter(new_v))
print(nb_bow_fv)

