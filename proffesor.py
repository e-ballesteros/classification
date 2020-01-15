#!/usr/bin/env python3
#Author: Mauro De Sanctis, PhD, University of Rome "Tor Vergata"

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from pmf_estimation_multivariate import pmf_multivariate
from mpl_toolkits import mplot3d	# This import registers the 3D projection, but is otherwise unused.

iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target

#print("Data shape =", data_matrix.shape)    #The Iris dataset is a 150x4 data matrix
#print("Data matrix =", data_matrix)
#print("Class vector =", class_vector)
f1 = plt.figure(1)
ax1 = plt.axes(projection='3d')
ax1.scatter(data_matrix[:, 0], data_matrix[:, 1], data_matrix[:, 2], c=class_vector, edgecolor='k', s=40)

########################################################################################################################
#List of n features-dimensional data points. Each row corresponds to a single data point
#We have selected two columns from the data matrix
#We may also select rows belonging to the same class
#data_samples = np.transpose(np.vstack((data_matrix[0:50, 0], data_matrix[0:50, 1])))   #1st class
#data_samples = np.transpose(np.vstack((data_matrix[50:100, 0], data_matrix[50:100, 1]))) #2nd class
#data_samples = np.transpose(np.vstack((data_matrix[100:150, 0], data_matrix[100:150, 1]))) #3rd class
data_samples = np.transpose(np.vstack((data_matrix[:, 0], data_matrix[:, 1])))  #All classes

#print("Data samples=", data_samples)
#print("Data samples shape=", data_samples.shape)

feature_1_min, feature_1_max, feature_1_std, feature_1_mean = data_samples[:, 0].min(), data_samples[:, 0].max(), data_samples[:, 0].std(), data_samples[:, 0].mean()
feature_2_min, feature_2_max, feature_2_std, feature_2_mean = data_samples[:, 1].min(), data_samples[:, 1].max(), data_samples[:, 1].std(), data_samples[:, 1].mean()
print("Feature 1 mean=", feature_1_mean)
print("Feature 2 mean=", feature_2_mean)

########################################################################################à
N_samples = 10
start_sample_1 = feature_1_mean - 2*feature_1_std
start_sample_2 = feature_2_mean - 2*feature_2_std
stop_sample_1 = feature_1_mean + 2*feature_1_std
stop_sample_2 = feature_2_mean + 2*feature_2_std
X_plot = np.linspace(start_sample_1, stop_sample_1, N_samples, endpoint=True)   #row vector
Y_plot = np.linspace(start_sample_2, stop_sample_2, N_samples, endpoint=True)   #row vector

data_plot_X, data_plot_Y = np.meshgrid(X_plot, Y_plot)      #Transform vectors to matrices through repetition
                                                            #Grids X,Y should be like: X=[[0, 1, 2,..., N], [0, 1, 2,..., N],...,[0, 1, 2,..., N]]
                                                            #and Y=[[0, 0, ..., 0], [1, 1, ...,1],...,[N, N,..., N]
print("data plot X=", data_plot_X)
data_plot_X_vectorized = data_plot_X.flatten()              #Vectorize the grid matrix data_plot_X
data_plot_Y_vectorized = data_plot_Y.flatten()              #Vectorize the grid matrix data_plot_Y
#print("data plot X vectorized=", data_plot_X_vectorized)

data_plot = np.transpose(np.vstack((data_plot_X_vectorized, data_plot_Y_vectorized)))
print("data plot=", data_plot)

bandwidthKDE = 0.4                #As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
kernelFunction = 'gaussian'     #Valid kernel functions are: ‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’
kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(data_samples)

kde_LogDensity_estimate = kde_object.score_samples(data_plot)
kde_estimate = np.exp(kde_LogDensity_estimate)
print("KDE estimate =", kde_estimate)

f2 = plt.figure(2)
ax2 = plt.axes(projection='3d')
ax2.plot_trisurf(data_plot[:, 0], data_plot[:, 1], kde_estimate, linewidth=0, antialiased=False)

#Transform an array of floats to an array of integers. For type conversion of numpy arrays
#int() cannot be used for numpy arrays, instead use .astype(int)
discretized_data_matrix = (10*data_matrix[:, 0:2]).astype(int)
unique_rows, pmf_vector = pmf_multivariate(discretized_data_matrix)
#print("discretized matrix=", discretized_data_matrix)
#print("pmf multivariate function=", pmf_multivariate(discretized_data_matrix))
#print("length of unique rows", len(unique_rows))
#print(sum(pmf_vector))          #Must return 1

plt.show()