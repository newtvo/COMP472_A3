import csv

from Model.model_tung import MNB_Model
from Model.preprocessing import *

X = get_features()
y = get_values()
vocabulary = get_original_vocabulary()
model = MNB_Model(X,y, vocabulary)
# def get_report(file):
#     tsv_file = open('../Dataset/covid_test_public.tsv.tsv', encoding="utf8")
#     read_tsv = csv.reader(tsv_file, delimiter="\t")
 #open file
# loop each row in testing file
text_count = 0
correct_count = 0
value_dict = {1: "yes", 0: "no"}
true_positive = 0
false_postive =  0
false_negative = 0
true_negative = 0
for row in test_file:
    result = "wrong"
     id = row[0]
     text = clean_text(row[1])
     value = row[2]
     prediction, yes_score, no_score  = model.predict(text)
     if value_dict[prediction] == value:
         correct_count += 1
        result = "correct"
    text_count +=1
    if(value_dict[prediction] == value == "yes"):
        true_positive += 1
    elif(prediction == 1 and value == "no"):
        false_postive += 1
    elif(prediction == 0 and value == "yes"):
        false_negative +=1
    else:
        true_negative +=1
    trace_file.write(id, value_dict[prediction], score, value, result )

accuracy = correct_count/text_count
yes_precision = true_positive/(true_positive + false_postive)
yes_recall = true_positive /(true_positive + false_negative)
no_precision = true_negative/(true_negative + false_negative)
no_recall = true_negative /(true_negative + false_postive)
yes_f1_measure = 2 * yes_precision * yes_recall / (yes_precision + yes_recall)
no_f1_measure = 2 * no_precision * no_recall / (no_precision + no_recall)
eval_file.write(accuracy,yes_precision, no_precision, yes_recall, no_recall, yes_f1_measure, no_f1_measure )

