#!/usr/bin/env python3


import numpy as np
from classifier_functions import features_characteristics
from classifier_functions import find_closest_value
from classifier_functions import split_into_classes


# Naive Bayes classifier function that takes a training data set, a class label vector for train dataset and a test
# dataset and returns a class label vector for the test dataset
def naive_bayes_classifier(data_training, class_vector, data_test):

    from estimation_pmf import estimate_pmf
    from sklearn.neighbors import KernelDensity

    # List of the kde_estimate of each feature of a certain class
    kde_estimate = []

    # Estimation of pmf of class_vector in class_pmf and storage of classes list in unique_class_list
    unique_class_list, class_pmf = estimate_pmf(class_vector)

    # List with the four possible mult_likelihoods lists of each class
    group_mult_likelihoods = []

    # List with the multiplication of likelihoods of a certain class
    individual_mult_likelihoods = []

    # List with the class vector that is returned as result
    class_label_vector = []

    # x plot of each feature
    x_plot = []

    # Split the dataset into subdatasets according to the classes
    group_subdatasets = split_into_classes(data_training, class_vector)

    n_samples = int(input('Introduce number of samples (Naive Bayes classifier): '))
    kernel_function = input('Introduce Kernel function you want to use (Naive Bayes classifier) '
                            '(gaussian, tophat, epanechnikov, exponential, linear or cosine): ')

    rows_data_training, columns_data_training = data_training.shape

    # Done for every subdataset of each class
    for i in range(0, len(group_subdatasets)):
        for j in range(0, columns_data_training):
            feature_min, feature_max, feature_std, feature_mean = features_characteristics(group_subdatasets[i][:, j])
            margin = feature_std * 2
            optimal_bandwidth = 1.06 * feature_std * np.power(rows_data_training, -1 / 5)

            kde_object = KernelDensity(kernel=kernel_function,
                                       bandwidth=optimal_bandwidth).fit(group_subdatasets[i][:, j].reshape(-1, 1))
            x_plot.append(np.linspace(feature_min - margin, feature_max + margin, n_samples)[:, np.newaxis])
            kde_logdensity_estimate = kde_object.score_samples(x_plot[j])

            # Append the kde_estimate of each feature of a certain class
            kde_estimate.append(np.exp(kde_logdensity_estimate))

        rows_data_test, columns_data_test = data_test.shape

        # Find the indexes of the 4 features that are most similar to the instance
        for k in range(0, rows_data_test):
            index_0 = find_closest_value(x_plot[0], data_test[k][0])
            index_1 = find_closest_value(x_plot[1], data_test[k][1])
            index_2 = find_closest_value(x_plot[2], data_test[k][2])
            index_3 = find_closest_value(x_plot[3], data_test[k][3])

            individual_mult_likelihoods.append(kde_estimate[0][index_0] *
                                               kde_estimate[1][index_1] *
                                               kde_estimate[2][index_2] *
                                               kde_estimate[3][index_3])

        group_mult_likelihoods.append(individual_mult_likelihoods)    # Stores in group the class mult_likelihood vector
        individual_mult_likelihoods = []
        x_plot = []

    results = []

    # For each row of the data_test corresponding to each sample of it
    for i in range(0, rows_data_test):
        # For each class of the dataset
        for j in range(0, len(unique_class_list)):
            results.append(group_mult_likelihoods[j][i]*class_pmf[j])  # Multiplication of likelihoods and probabilities
        class_label_vector.append(np.argmax(results))
        results = []

    return class_label_vector
