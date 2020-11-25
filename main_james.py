import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from Model.naive_bayes import NaiveBayes

# reading the training dataset
training_set = pd.read_csv('./Dataset/covid_training.tsv', sep = '\t')

# getting training set examples labels
y_train = training_set['q1_label'].values
x_train = training_set['text'].values
print("Unique Classes: ", np.unique(y_train))
print("Total Number of Training Examples: ", x_train.shape)

# train_test_split to have an unbiased evaluation and validation process
train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
classes=np.unique(train_labels)


# Training phase....

# nb=NaiveBayes(classes)
# print ("------------------Training In Progress------------------------")
# print ("Training Examples: ",train_data.shape)
# nb.train(train_data,train_labels)
# print ('------------------------Training Completed!')
#
# # Testing phase
#
# pclasses=nb.test(test_data)
# test_acc=np.sum(pclasses==test_labels)/float(test_labels.shape[0])
# print ("Test Set Examples: ",test_labels.shape[0])
# print ("Test Set Accuracy: ",test_acc)
#
# # Loading the kaggle test dataset
# test=pd.read_csv('./Dataset/covid_test_public.tsv',sep='\t')
# # Xtest=test.
# print(test.columns[1])
#
# # #generating predictions....
# # pclasses=nb.test(Xtest)
