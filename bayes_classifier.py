#!/usr/bin/env python3

import numpy as np


def features_characteristics(feature_vector):
    return feature_vector.min(), feature_vector.max(), feature_vector.std(), feature_vector.mean()


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
        group_datasets.append(individual_dataset)

    return group_datasets


# The dataset must have instances grouped
def bayes_classifier(data_training, class_vector, data_test):
    from estimation_pmf import estimate_pmf
    from sklearn.neighbors import KernelDensity

    class_list, class_pmf = estimate_pmf(class_vector)

    # I think this is not necessary since we don't want to take only 2 features
    #data_samples = np.transpose(np.vstack((data_training[:, 0], data_training[:, 1])))  # All classes

    #data_samples_0 = data_training[0:50, :]
    #data_samples_0 = np.transpose(np.vstack((data_training[0:50, 0], data_training[0:50, 1])))   #1st class
    #data_samples_1 = np.transpose(np.vstack((data_training[50:100, 0], data_training[50:100, 1]))) #2nd class
    #data_samples_2 = np.transpose(np.vstack((data_training[50:150, 0], data_training[50:150, 1]))) #3rd class
    # data_samples = np.transpose(np.vstack((data_matrix[100:150, 0], data_matrix[100:150, 1]))) #3rd class

    # print("Data samples=", data_samples)
    # print("Data samples shape=", data_samples.shape)

    # Split the dataset into subdatasets according to the classes
    group_subdatasets = split_into_classes(data_training, class_vector)

    N_samples = int(input('Introduce number of samples: '))
    kernel_function = input('Introduce Kernel function you want to use (gaussian, tophat, epanechnikov,'
                            'exponential, linear or cosine): ')
    bandwidth_kde = float(input('Introduce bandwidth : '))

    for i in range(0, len(group_subdatasets)):
        feature_0_min, feature_0_max, feature_0_std, feature_0_mean = features_characteristics(group_subdatasets[i][:, 0])
        feature_1_min, feature_1_max, feature_1_std, feature_1_mean = features_characteristics(group_subdatasets[i][:, 1])
        feature_2_min, feature_2_max, feature_2_std, feature_2_mean = features_characteristics(group_subdatasets[i][:, 2])
        feature_3_min, feature_3_max, feature_3_std, feature_3_mean = features_characteristics(group_subdatasets[i][:, 3])

        print("Feature 0 mean=", feature_0_mean)
        print("Feature 1 mean=", feature_1_mean)
        print("Feature 2 mean=", feature_2_mean)
        print("Feature 3 mean=", feature_3_mean)

        start_sample_0, stop_sample_0 = start_and_stop(feature_0_mean, feature_0_std)
        start_sample_1, stop_sample_1 = start_and_stop(feature_1_mean, feature_1_std)
        start_sample_2, stop_sample_2 = start_and_stop(feature_2_mean, feature_2_std)
        start_sample_3, stop_sample_3 = start_and_stop(feature_3_mean, feature_3_std)

        plot_0 = np.linspace(start_sample_0, stop_sample_0, N_samples, endpoint=True)  # row vector
        plot_1 = np.linspace(start_sample_1, stop_sample_1, N_samples, endpoint=True)  # row vector
        plot_2 = np.linspace(start_sample_2, stop_sample_2, N_samples, endpoint=True)  # row vector
        plot_3 = np.linspace(start_sample_3, stop_sample_3, N_samples, endpoint=True)  # row vector

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
        kde_estimate = np.exp(kde_log_density_estimate)
        print("KDE estimate =", kde_estimate)