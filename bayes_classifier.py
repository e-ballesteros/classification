#!/usr/bin/env python3

import numpy as np


def features_characteristics(feature_vector):
    return feature_vector.min(), feature_vector.max(), feature_vector.std(), feature_vector.mean()


def start_and_stop(feature_mean, feature_std):
    return feature_mean - 2 * feature_std, feature_mean + 2 * feature_std


# Returns a list of the different subdatasets splitted by classes
def split_into_classes(data_matrix, class_vector):

    unique_class_list = np.unique(class_vector, return_counts=False)   # Returns unique elements in class_vector

    rows, columns = data_matrix.shape

    group_datasets = []                      # List in which the different subdatasets of classes will be allocated
    individual_dataset = []                  # List in which a certain dataset is stored

    for i in range(0, len(unique_class_list)):                  # It is splitted into len(classes_list) number of matrix
        #individual_dataset = np.empty([][columns])
        for j in range(0, len(class_vector)):                   # All the elements within class_vector
            if class_vector[j] == unique_class_list[i]:         # Class_vector element compared with a certain class
                individual_dataset.append(data_matrix[j])
        group_datasets.append(np.array(individual_dataset))     # Append the subdataset as numpy array

    return group_datasets


# Returns the index of the element of the array most similar to the number
def find_closest_value(array, number):
    return (np.abs(array - number)).argmin()


def bayes_classifier(data_training, class_vector, data_test):
    from estimation_pmf import estimate_pmf
    from sklearn.neighbors import KernelDensity

    # List of the kde_estimate of each subdataset corresponding to each class
    kde_estimate = []

    # Estimation of pmf of class_vector in class_pmf and storage of classes list in unique_class_list
    unique_class_list, class_pmf = estimate_pmf(class_vector)

    # List with the four possible likelihoods lists for each class
    group_likelihoods = []

    # List with an individual likelihood
    individual_likelihood = []

    # List with the class vector that is returned as result
    class_label_vector = []

    # Split the dataset into subdatasets according to the classes
    group_subdatasets = split_into_classes(data_training, class_vector)

    n_samples = int(input('Introduce number of samples: '))
    kernel_function = input('Introduce Kernel function you want to use (gaussian, tophat, epanechnikov,'
                            'exponential, linear or cosine): ')
    bandwidth_kde = float(input('Introduce bandwidth : '))

    for i in range(0, len(group_subdatasets)):
        feature_0_min, feature_0_max, feature_0_std, feature_0_mean = features_characteristics(group_subdatasets[i][:, 0])
        feature_1_min, feature_1_max, feature_1_std, feature_1_mean = features_characteristics(group_subdatasets[i][:, 1])
        feature_2_min, feature_2_max, feature_2_std, feature_2_mean = features_characteristics(group_subdatasets[i][:, 2])
        feature_3_min, feature_3_max, feature_3_std, feature_3_mean = features_characteristics(group_subdatasets[i][:, 3])

        start_sample_0, stop_sample_0 = start_and_stop(feature_0_mean, feature_0_std)
        start_sample_1, stop_sample_1 = start_and_stop(feature_1_mean, feature_1_std)
        start_sample_2, stop_sample_2 = start_and_stop(feature_2_mean, feature_2_std)
        start_sample_3, stop_sample_3 = start_and_stop(feature_3_mean, feature_3_std)

        plot_0 = np.linspace(start_sample_0, stop_sample_0, n_samples, endpoint=True)  # row vector
        plot_1 = np.linspace(start_sample_1, stop_sample_1, n_samples, endpoint=True)  # row vector
        plot_2 = np.linspace(start_sample_2, stop_sample_2, n_samples, endpoint=True)  # row vector
        plot_3 = np.linspace(start_sample_3, stop_sample_3, n_samples, endpoint=True)  # row vector

        data_plot_0, data_plot_1, data_plot_2, data_plot_3 = np.meshgrid(plot_0, plot_1, plot_2, plot_3)  # Transform vectors to matrices through repetition

        data_plot_0_vectorized = data_plot_0.flatten()  # Vectorize the grid matrix data_plot_0
        data_plot_1_vectorized = data_plot_1.flatten()  # Vectorize the grid matrix data_plot_1
        data_plot_2_vectorized = data_plot_2.flatten()  # Vectorize the grid matrix data_plot_2
        data_plot_3_vectorized = data_plot_3.flatten()  # Vectorize the grid matrix data_plot_3

        data_plot = np.transpose(np.vstack((data_plot_0_vectorized,
                                            data_plot_1_vectorized,
                                            data_plot_2_vectorized,
                                            data_plot_3_vectorized)))

        kde_object = KernelDensity(kernel=kernel_function, bandwidth=bandwidth_kde).fit(data_training)

        kde_log_density_estimate = kde_object.score_samples(data_plot)
        kde_estimate.append(np.exp(kde_log_density_estimate))

        rows_data_test, columns_data_test = data_test.shape

        ############################################### THIS IS WRONG ############################################
        # Find the indexes of the 4 features most similar to the data_test row (instance)
        for j in range(0, rows_data_test):
            index_0 = find_closest_value(plot_0, data_test[j][0])
            index_1 = find_closest_value(plot_1, data_test[j][1])
            index_2 = find_closest_value(plot_2, data_test[j][2])
            index_3 = find_closest_value(plot_3, data_test[j][3])
            index_0123 = int(str(index_0) + str(index_1) + str(index_2) + str(index_3))
            individual_likelihood.append(kde_estimate[i][index_0123])
        ############################################### THIS IS WRONG ############################################

        group_likelihoods.append(individual_likelihood)             # Stores in group the class likelihood vector
        individual_likelihood = []

    results = []

    for i in range(0, rows_data_test):
        for j in range(0, len(unique_class_list)):
            results.append(group_likelihoods[j][i]*class_pmf[j])    # Multiplication of likelihoods and probabilities
        class_label_vector.append(np.argmax(results))
        results = []

    return class_label_vector
