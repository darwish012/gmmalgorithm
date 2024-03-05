# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:39:36 2023

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression



matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')
#################step one########################################################
df = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\GUC\Semester 7\(CSEN707) Analysis and Design of Algorithms\lecs&tuts\dataset.csv')

#column_headers = list(df.columns.values)
#your_dataframe_sampled = df.sample(frac=0.01, random_state=42)
def normalize_data(data):
    # Assuming data is a DataFrame
    return (data - data.mean()) / data.std()

def gaussian(x, mean, cov):
    n = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean))
    coef = 1 / ((2 * np.pi) ** (n / 2) * det_cov ** 0.5)
    return coef * np.exp(exponent)

#initializes the parameters of the GMM model, including the mean vector and covariance matrix for each cluster.
def initialize_parameters(data, k):
    n_samples, n_features = data.shape
    means = data.sample(k).values
    covariances = [np.cov(data.values.T) for _ in range(k)]
    weights = np.ones(k) / k
    return means, covariances, weights



# def initialize_parameters(data, k):
#     n_samples, n_features = data.shape
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(data)
#     means = kmeans.cluster_centers_

#     # Use k-means to initialize covariances
#     labels = kmeans.predict(data)
#     covariances = [np.cov(data[labels == i].T) for i in range(k)]

#     # Initialize weights equally
#     weights = np.ones(k) / k
#     return means, covariances, weights

#expectation step calculate the expected likelihood function given parmeters
#computes the responsibilities of each sample for each cluster.
def expectation(data, means, covariances, weights):
    """
    Calculate the responsibilities using vectorized operations for a Gaussian Mixture Model (GMM).

    Parameters:
    - data: The input data matrix (n_samples x n_features).
    - means: The mean vectors for each cluster (k x n_features).
    - covariances: The covariance matrices for each cluster (k x n_features x n_features).
    - weights: The weights of each cluster (k,).

    Returns:
    - responsibilities: Matrix of responsibilities (n_samples x k).
    """
    n_samples, n_features = data.shape
    k = len(means)

    # Initialize matrix to store responsibilities for each sample and each cluster
    responsibilities = np.zeros((n_samples, k))

    # Calculate the responsibilities using vectorized operations
    for i in range(k):
        # Calculate the probability density function (PDF) for each sample in cluster i
        pdf_values = multivariate_normal.pdf(data, means[i], covariances[i])

        # Multiply PDF values by the weight of cluster i
        responsibilities[:, i] = weights[i] * pdf_values

    # Normalize responsibilities
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    return responsibilities

#updates the parameters of the GMM model using the responsibilities computed in the expectation function to maximise the loglikhood func.
def maximization(data, responsibilities):
    """
    Update the parameters (means, covariances, and weights) of a Gaussian Mixture Model (GMM) using the responsibilities.

    Parameters:
    - data: The input data matrix (n_samples x n_features).
    - responsibilities: Matrix of responsibilities (n_samples x k).

    Returns:
    - means: Updated mean vectors for each cluster (k x n_features).
    - covariances: Updated covariance matrices for each cluster (k x n_features x n_features).
    - weights: Updated weights of each cluster (k,).
    """
    n_samples, n_features = data.shape
    k = responsibilities.shape[1]

    # Initialize parameters
    means = np.zeros((k, n_features))
    covariances = [np.zeros((n_features, n_features)) for _ in range(k)]
    weights = np.zeros(k)

    for i in range(k):
        # Update weights
        weights[i] = np.sum(responsibilities[:, i])

        # Update means
        means[i] = np.dot(responsibilities[:, i], data) / weights[i]

        # Update covariances
        diff = data - means[i]
        covariances[i] = np.dot((responsibilities[:, i][:, np.newaxis] * diff).T, diff) / weights[i]

    # Normalize weights
    weights /= n_samples

    return means, covariances, weights


#takes the input data and the number of clusters as input, and returns the cluster assignments for each sample.
def gmm(data, k, n_iterations=25):
    #  normalize data

    data = normalize_data(data)
    # Initialize parameters

    means, covariances, weights = initialize_parameters(data, k)
   
    #O(KNiterationf)
    for iteration in range(n_iterations):
       # Expectation step
        responsibilities = expectation(data, means, covariances, weights)
        # Maximization step
        means, covariances, weights = maximization(data, responsibilities)
    # Assign clusters based on the maximum responsibility
    clusters = np.argmax(responsibilities, axis=1)
    return clusters

def gmm_without_normalising(data, k, n_iterations=100, tol=1e-15):
    means, covariances, weights = initialize_parameters(data, k)
    prev_log_likelihood = float('-inf')

    for iteration in range(n_iterations):
        responsibilities = expectation(data, means, covariances, weights)
        means, covariances, weights = maximization(data, responsibilities)

        # Calculate log-likelihood for the current iteration
        log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))

        # Check for convergence
        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        prev_log_likelihood = log_likelihood
    # Assign clusters based on the maximum responsibility
    clusters = np.argmax(responsibilities, axis=1)
    return clusters



# Experiment
Ns = np.arange(0.01, 1.01, 0.01)
Ks = [3, 5]  # Different values of K

#store the running times of the GMM algorithm for different values of the number of samples and the number of clusters.
runtimes = []

for K in Ks:
    for N in Ns:
        data =df.sample(frac=N, random_state=42)
        
        start_time = time.time()
        clusters = gmm(data, K)
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append((N, K, runtime))

# Plotting
columns = ['Number_of_Samples', 'Number_of_Clusters', 'Runtime']
runtimes_df = pd.DataFrame(runtimes, columns=columns)

# Save the DataFrame to a CSV file
runtimes_df.to_csv('runtimes.csv', index=False)
for K in Ks:
    N_values = [entry[0] for entry in runtimes if entry[1] == K]
    runtime_values = [entry[2] for entry in runtimes if entry[1] == K]
    
    plt.plot(N_values, runtime_values, label=f'K={K}')

plt.xlabel('Number of Samples (N)')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.show()
plt.figure()
Ns_values = {}
runtime_values = {}
for K in Ks:
    N_values = [entry[0] for entry in runtimes if entry[1] == K]
    runtime_values[K] = [entry[2] for entry in runtimes if entry[1] == K]
    Ns_values[K] = N_values

# Plotting
plt.figure()
for K in Ks:
    N_values = Ns_values[K]
    runtime_values_K = runtime_values[K]

    # Smooth the lines using interpolation
    N_smooth = np.linspace(min(N_values), max(N_values), 300)
    runtime_smooth = make_interp_spline(N_values, runtime_values_K, k=3)(N_smooth)

    plt.plot(N_smooth, runtime_smooth, label=f'K={K}')

plt.xlabel('Number of Samples (N)')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.show()

# Linear regression
plt.figure()
for K in Ks:
    N_values = np.array(Ns_values[K]).reshape(-1, 1)
    runtime_values_K = runtime_values[K]

    # Fit linear regression model
    model = LinearRegression().fit(N_values, runtime_values_K)

    # Predict runtimes using the model
    predicted_runtimes = model.predict(N_values)

    # Plot the original data points
    plt.scatter(N_values, runtime_values_K, label=f'Actual Runtimes (K={K})')

    # Plot the line of best fit
    plt.plot(N_values, predicted_runtimes, label=f'Line of Best Fit (K={K})', linestyle='--')

plt.xlabel('Number of Samples (N)')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.show()

####################################################################### Milestone 1 end##########################################################


data_normalized = normalize_data(df)
pca = PCA(n_components=2)
pca_df = pca.fit_transform(data_normalized)
pca_df=pd.DataFrame(pca_df)
# pca_df_plot=pca.fit_transform(data)

#store the running times of the GMM algorithm for different values of the number of samples and the number of clusters.
# Plot the samples on the x-y coordinates (2D features) colored according to their cluster one plot for each k

cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']

z=Ks[0]

# ...

# ...
# ...

for K in Ks:
    # Use your sampled data here
    if K != z:
        pca_df = pca_df.drop(columns=['Cluster'])

    clusters = gmm_without_normalising(pca_df, K)
    cluster_sizes = np.bincount(clusters)
    total_samples = len(clusters)
    percentages = cluster_sizes / total_samples * 100
    pca_df['Cluster'] = clusters

    # Plot the samples in 2D PCA space colored according to their cluster
    plt.figure()
    for i in range(K):
        cluster_data = pca_df[pca_df['Cluster'] == i]
        plt.scatter(cluster_data[0], cluster_data[1], c=cluster_colors[i], label=f'Cluster {i}')

    plt.title(f'GMM Clustering Results (K={K})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # Annotate percentages above and to the right of the plot
    text_content = f'K={K}\n'
    for idx, label in enumerate(['S', 'M', 'L', 'XL', 'XXL']):
        if idx < len(percentages):
            text_content += f'{label}: {percentages[idx]:.2f}%\n'

    plt.text(
        1.02, 1,
        text_content,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.show()
