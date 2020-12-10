from Model.model import MNB_Model
from Model.preprocessing import *

def fv_run():
    X = get_features()
    y = get_values()
    filtered_vocabulary = get_filtered_vocalbulary()
    model_fv = MNB_Model(X, y, filtered_vocabulary)

    # open file
    trace_fv_filename = '../Output/trace_nb_bow_fv.txt'
    eval_fv_filename = '../Output/eval_nb_bow_fv.txt'
    trace_fv_file = open(trace_fv_filename, 'w')
    eval_fv_file = open(eval_fv_filename, 'w')

    # loop each row in testing file

    text_count = 0
    correct_count = 0
    value_dict = {1: "yes", 0: "no"}
    score = 0
    true_positive = 0
    false_postive = 0
    false_negative = 0
    true_negative = 0
    result = "wrong"
    for row in get_test_file():
        id = row[0]
        text = clean_text(row[1])
        value = row[2]
        prediction, yes_score, no_score = model_fv.predict(text)

        if value_dict[prediction] == value:
            correct_count += 1
            score = yes_score
            result = "correct"
        else:
            score = no_score
            result = "wrong"

        text_count += 1

        if value_dict[prediction] == value == "yes":
            true_positive += 1
        elif (prediction == 1 and value == "no"):
            false_postive += 1
        elif (prediction == 0 and value == "yes"):
            false_negative += 1
        else:
            true_negative += 1
        print(id)
        trace_fv_file.write("{0}  {1}  {2}  {3}  {4} \n".format(id, value_dict[prediction], score, value, result))

    # metrics
    accuracy = correct_count / text_count
    yes_precision = true_positive / (true_positive + false_postive)
    yes_recall = true_positive / (true_positive + false_negative)
    no_precision = true_negative / (true_negative + false_negative)
    no_recall = true_negative / (true_negative + false_postive)

    yes_f1_measure = 2 * yes_precision * yes_recall / (yes_precision + yes_recall)
    no_f1_measure = 2 * no_precision * no_recall / (no_precision + no_recall)

    eval_fv_file.write("{0}\n".format(accuracy))
    eval_fv_file.write("{0}  {1}\n".format(yes_precision, no_precision))
    eval_fv_file.write("{0}  {1}\n".format(yes_recall, no_recall))
    eval_fv_file.write("{0}  {1}\n".format(yes_f1_measure, no_f1_measure))

    trace_fv_file.close()
    eval_fv_file.close()