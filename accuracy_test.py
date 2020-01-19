#!/usr/bin/env python3


from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from bayes_classifier import bayes_classifier
from naive_bayes_classifier import naive_bayes_classifier


# Returns the rate of equal values in two class vectors
def compare_class_vectors(class_vector_a, class_vector_b):
    matches = 0
    for i in range(0, len(class_vector_a)):
        if class_vector_a[i] == class_vector_b[i]:
            matches += 1
    return matches/len(class_vector_a)


# Bayes_classifier can be used with every dataset. In this file it is tested with the iris dataset
iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target

class_vector_train = []     # Class_vector of the train_matrix
class_vector_test = []      # Class_vector of the test_matrix

print('Data matrix: ', data_matrix)

test_size = float(input('Introduce proportion of the dataset to be included in test split from 0 to 1: '))

# List with the numbers of the rows of the dataset
y = np.linspace(0, 149, 150, endpoint=True)
print('y: ', y)

# X_train and X_test are the matrices splitted and y_train and y_test are the rows selected in each case
X_train, X_test, y_train, y_test = train_test_split(data_matrix, y, test_size=test_size)

print('X_train: ', X_train)
print('X_test: ', X_test)
print('y_train: ', y_train)
print('y_test: ', y_test)

for i in range(0, len(y_train)):
    # y_train[i] is the index of the class_vector element we need to append
    class_vector_train.append(class_vector[int(y_train[i])])

for i in range(0, len(y_test)):
    # y_test[i] is the index of the class_vector element we need to append
    class_vector_test.append(class_vector[int(y_train[i])])

print('class_vector_train: ', class_vector_train)

class_label_vector_bayes = bayes_classifier(X_train, class_vector_train, X_test)
class_label_vector_naive = naive_bayes_classifier(X_train, class_vector_train, X_test)

accuracy_Bayes = compare_class_vectors(class_label_vector_bayes, class_vector_test)
accuracy_Naive_Bayes = compare_class_vectors(class_label_vector_naive, class_vector_test)

print('class_vector_test: ', class_vector_test)
print('class_label_vector Bayes: ', class_label_vector_bayes)
print('class_label_vector Naive Bayes: ', class_label_vector_naive)

print('Accuracy of Bayes is: ', accuracy_Bayes)
print('Accuracy of Naive Bayes is: ', accuracy_Naive_Bayes)

