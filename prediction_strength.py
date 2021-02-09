
# This script was borrowed from Eryk Lewinson's article on Towards Data Science, and was
# posted on his GitHub repository:
# https://github.com/erykml/medium_articles/blob/master/Machine%20Learning/prediction_strength.ipynb

import sys

import numpy as np
from scipy.spatial import distance


def get_closest_centroid(obs, centroids):
    '''
    Function for retrieving the closest centroid to the given observation
    in terms of the Euclidean distance.

    Parameters
    ----------
    obs : array
        An array containing the observation to be matched to the nearest centroid
    centroids : array
        An array containing the centroids

    Returns
    -------
    min_centroid : array
        The centroid closes to the obs
    '''
    min_distance = sys.float_info.max
    min_centroid = 0

    for c in centroids:
        dist = distance.euclidean(obs, c)
        if dist < min_distance:
            min_distance = dist
            min_centroid = c

    return min_centroid


def get_prediction_strength(k, train_centroids, x_test, test_labels):
    '''
    Function for calculating the prediction strength of clustering

    Parameters
    ----------
    k : int
        The number of clusters
    train_centroids : array
        Centroids from the clustering on the training set
    x_test : array
        Test set observations
    test_labels : array
        Labels predicted for the test set

    Returns
    -------
    prediction_strength : float
        Calculated prediction strength
    '''
    n_test = len(x_test)

    # populate the co-membership matrix
    D = np.zeros(shape=(n_test, n_test))
    for x1, l1, c1 in zip(x_test, test_labels, list(range(n_test))):
        for x2, l2, c2 in zip(x_test, test_labels, list(range(n_test))):
            if tuple(x1) != tuple(x2):
                if tuple(get_closest_centroid(x1, train_centroids)) == tuple(get_closest_centroid(x2, train_centroids)):
                    D[c1, c2] = 1.0

    # calculate the prediction strengths for each cluster
    ss = []
    for j in range(k):
        s = 0
        examples_j = x_test[test_labels == j, :].tolist()
        n_examples_j = len(examples_j)
        for x1, l1, c1 in zip(x_test, test_labels, list(range(n_test))):
            for x2, l2, c2 in zip(x_test, test_labels, list(range(n_test))):
                if tuple(x1) != tuple(x2) and l1 == l2 and l1 == j:
                    s += D[c1, c2]
        ss.append(s / (n_examples_j * (n_examples_j - 1)))

    prediction_strength = min(ss)

    return prediction_strength

