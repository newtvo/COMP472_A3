import csv

#=========================================================
tsv_file = open('../Dataset/covid_training.tsv')
read_tsv = csv.reader(tsv_file, delimiter="\t")
# from the description, we should label this as V which will be used as features
vocabulary = []
values = []
V = []
for i, row in enumerate(read_tsv):
    #skip the head
    if(i > 0):
        #extract each value from column text and use casefold() to change the values in lower case
        values = row[1].casefold().split()
        for words in values:
            V.append(words)

print (V)
