#!/usr/bin/env python3

import numpy as np
from scipy import spatial                            # Needed to perform a search of the closest vector within a matrix


# Minimum, maximum, std and mean of a certain feature
def features_characteristics(feature_vector):
    return feature_vector.min(), feature_vector.max(), feature_vector.std(), feature_vector.mean()


# Start sample and stop sample for the plot
def start_and_stop(feature_mean, feature_std):
    return feature_mean - 2 * feature_std, feature_mean + 2 * feature_std


# Returns a list of the different subdatasets splitted by classes
def split_into_classes(data_matrix, class_vector):

    unique_class_list = np.unique(class_vector, return_counts=False)   # Returns unique elements in class_vector

    group_datasets = []                      # List in which the different subdatasets of classes will be allocated
    individual_dataset = []                  # List in which a certain dataset is stored

    for i in range(0, len(unique_class_list)):                  # It is splitted into len(classes_list) number of matrix
        for j in range(0, len(class_vector)):                   # All the elements within class_vector
            if class_vector[j] == unique_class_list[i]:         # Class_vector element compared with a certain class
                individual_dataset.append(data_matrix[j])
        group_datasets.append(np.array(individual_dataset))     # Append the subdataset as numpy array
        individual_dataset = []

    return group_datasets


# Returns the index of the element of the array most similar to the number
def find_closest_vector(matrix, array_a):
    tree = spatial.KDTree(matrix)
    distance, index = tree.query(array_a)
    return index


# Returns the index of the element of the array most similar to the number
def find_closest_value(array, number):
    return (np.abs(array - number)).argmin()
