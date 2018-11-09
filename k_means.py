import random
import numpy as np
from scipy.spatial import distance

class KMeans:

    def __init__(self, K):
        self.K = K
        self.centroids = np.zeros( (self.K, 2) )

    """
    Initializes the centroids for the k-means algorithm based on given data,
    choosing randomly within the bounds of the data.

    Parameters
    ----------
    X : Data given to make the random centroids from.
    """
    def initialize_centroids(self, X):
        idx = np.random.randint(len(X), size=self.K)

        self.centroids = X[idx, :]

    """
    Return the index for the closest centroid to the point p.

    Parameters
    ----------
    p : A data point to compute the closest centroid to.
    """
    def closest_centroid(self, p):
        return np.argmin( [np.sqrt(np.sum((c-p)**2)) for c in self.centroids] )

    """
    Computes the k-means clustering algorithm, moving the centroids.

    Parameters
    ----------
    X : The data to compute the algorithm on.
    """
    def fit(self, X):
        #randomly initialise centroids
        self.initialize_centroids(X)

        for _ in range(100):

            #assign the centroids to the data
            assigned_c = np.array( [self.closest_centroid(p) for p in X] )

            #move the centroids
            self.centroids = np.array([X[assigned_c == k].mean(axis = 0)
                                      for k in range(self.K)])

    """
    Predict clusters of a given dataset based on the closest distance to
    centroids.

    Parameters
    ----------
    X : The data to predict clusters for.
    """
    def predict(self, X):
        return np.array( [self.closest_centroid(p) for p in X] )

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=1000, centers=8,
                       cluster_std=0.60, random_state=1)

k_means = KMeans(8)
k_means.fit(X)
y_kmeans = k_means.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(k_means.centroids[:, 0], k_means.centroids[:, 1], c='black', s=200, alpha=0.7);
print(k_means.centroids)
plt.show()
