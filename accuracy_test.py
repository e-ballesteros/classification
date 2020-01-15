#!/usr/bin/env python3


from sklearn import datasets

iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target

print("Data shape =", data_matrix.shape)    #The Iris dataset is a 150x4 data matrix
print("Data matrix =", data_matrix)
print("Class vector =", class_vector)

