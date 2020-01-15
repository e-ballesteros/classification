#!/usr/bin/env python3
#Author: Mauro De Sanctis, PhD, University of Rome "Tor Vergata"

import numpy as np

# data_matrix has 'rows' row vector samples and 'columns' features

def pmf_multivariate(data_matrix):
    rows, columns = data_matrix.shape       #returns the number of rows and columns of the data_matrix
    unique_rows_array, pmf_vector = np.unique(data_matrix, axis=0, return_counts=True)  #the parameter axis=0 allows to count the unique rows
    return unique_rows_array, pmf_vector/rows       #To obtain the probability, the count must be normalized to the total count of samples


