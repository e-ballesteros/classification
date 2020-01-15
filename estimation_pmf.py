#!/usr/bin/env python3

import numpy as np


# Function that computes a pmf_vector given a list of samples
def estimate_pmf(samples):
    n = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)   # Counts unique elements in samples list
    return samples_list, pmf_vector/n                                   # Returns pmf vector and the samples list
