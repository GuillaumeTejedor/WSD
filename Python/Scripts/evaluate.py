from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import itertools
import numpy as np

def internal(X, cluster_labels, metric="euclidean", n_iter_hopkins=100, snorkel_labels=None):

    """
    Evaluate clusters by differents internal quality metrics.
    :param X: 2D array that represent either features for each sample or a measure matrix.
    :param cluster_labels: 1D array of clusters labels for each patient.
    :param metric: The metric to use when calculating distance between instances in a feature array.
    :param n_iter_hopkins: Number of times to compute Hopkins statistic before to compute avg and std.
    :param snorkel_labels: 1D array of snorkel labels for each pair of patients.
    :return: silhouette avg, silhouette std, hopkins avg, hopkins std, clusters distribution
    """
    m = int(np.ceil(0.1 * len(X)))
    max_value = np.max(X)
    hopkins_values = [compute_hopkins(X, m=m, low=0, high=max_value, metric=metric) for i in range(n_iter_hopkins)]

    silhouette_avg, silhouette_std = compute_silhouette(X, cluster_labels, metric=metric)
    hopkins_avg, hopkins_std = np.average(hopkins_values), np.std(hopkins_values)
    std_dist = compute_clusters_distribution(cluster_labels)

    if metric=="euclidean":
        ch_score = calinski_harabasz_score(X, cluster_labels)
    else:
        ch_score = -1.0

    if snorkel_labels is not None:
        matching_percentage = compute_matching_percentage(snorkel_labels, cluster_labels)
    else:
        matching_percentage = -1.0

    return silhouette_avg, silhouette_std, ch_score, hopkins_avg, hopkins_std, std_dist, matching_percentage

def compute_matching_percentage(snorkel_labels, cluster_labels):
    """
    Compute percentage of matching pairs of patients between Snorkel and clustering method 
    (i.e. are pairs together based on Snorkel when they are clustered together).

    :param snorkel_labels: 1D array of snorkel labels for each pair of patients.
    :param cluster_labels: 1D array of clusters labels for each patient.
    :return: Percentage of matching pairs between Snorkel and clustering algorithm.
    """

    # Prepare labels of pairs to specify if patients are in same cluster or not
    n = len(cluster_labels)
    pairs = list(itertools.combinations(range(n), 2))
    pairwise_list = [(i, j, int(cluster_labels[i] != cluster_labels[j])) for i, j in pairs]
    pairwise_array = np.array(pairwise_list)
    cluster_label_pairs = pairwise_array[:,-1]
    
    labels_state_clustering = cluster_label_pairs[snorkel_labels != -1]
    labels_state_snorkel = snorkel_labels[snorkel_labels != -1]

    matching_count = np.sum(labels_state_clustering == labels_state_snorkel)
    total_labels = labels_state_snorkel.size
    matching_percentage = (matching_count / total_labels) * 100
    return matching_percentage

def compute_silhouette(X, cluster_labels, metric="euclidean"):
    
    """
    Compute silhouette average and standard deviation from a 2D 
    array where each sample is labeled.

    :param X: 2D array that represent either features for each sample or a precomputed matrix.
    :param cluster_labels: 1D array of cluster cluster_labels.
    :param metric: The metric to use when calculating distance between instances in a feature array.
    :return: silhouette average, silhouette std
    """

    silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

    silhouette_avg = np.mean(silhouette_values)
    silhouette_std = np.std(silhouette_values)

    return silhouette_avg, silhouette_std

def compute_hopkins(X, m=5, low=0, high=1, metric="minkowski"):

    """
    Compute Hopkins statistic.

    :param X: 2D array that represent either features for each sample or a measure matrix.
    :param m: Integer that represent the number of data points to create for uniform random generation. 
    It has to be lesser than length of X.
    :param low: Real value that represent minimum thresold for uniform random values generation.
    :param max: Real value that represent maximum thresold for uniform random values generation.
    :param metric: Metric used to compute separation between individuals.
    :return: Hopkins statistic
    """

    if metric=="precomputed":

        # Define the number of additional rows/columns
        n_new = m
        #print("n_new=",n_new)
        # Generate new indexes
        new_indexes = [-(i+1) for i in range(n_new)]
        all_indexes = X.index.tolist() + new_indexes
        #print("X=",X)
        #print("X_indexes=", X.index.tolist())
        #print("X_all_indexes=", all_indexes)
        # Expand the DataFrame to include new rows and columns
        expanded_X = X.reindex(index=all_indexes, columns=all_indexes, fill_value=np.nan)
        #print("expanded_X=", expanded_X)
        # Fill new cells with random values between low and high in a symmetric way
        for new_label in new_indexes:
            for existing_label in all_indexes:
                if new_label != existing_label:
                    random_value = np.random.uniform(low=low, high=high)
                    expanded_X.loc[new_label, existing_label] = random_value
                    expanded_X.loc[existing_label, new_label] = random_value
        #print("expanded_X_filled=", expanded_X)
        # Set diagonal to 0 to maintain distance matrix properties
        np.fill_diagonal(expanded_X.values, 0)
                        
        indexes = expanded_X.index
        negative_indexes = indexes[indexes < 0].tolist()
        positive_indexes = indexes[indexes >= 0].tolist()
        # Compute u_distances
        Y = expanded_X.drop(positive_indexes, axis=0).drop(negative_indexes, axis=1)
        #print("Y=",Y)
        u_distances = np.array(Y.min(axis=1))
        #print("u_distances=",u_distances)
        # Compute w_distances
        #print("X=",X)
        n, d = X.shape
        #print("(n,m)=",(n,m))
        random_indices = np.random.choice(n, size=m, replace=False)
        #print("random_indices=", random_indices)
        X_tilt_indexes = X.index[random_indices]
        #print("X_tilt_indexes=", X_tilt_indexes)
        X_tilt = X.loc[X_tilt_indexes].drop(X_tilt_indexes, axis=1)
        #print("X_tilt=", X_tilt)
        w_distances = np.array(X_tilt.min(axis=1))
        #print("w_distances=", w_distances)
    else: # X is a matrix with n samples and is d dimensional
        X = np.array(X)
        n, d = X.shape
        random_indices = np.random.choice(n, size=m, replace=False)
        X_tilt = X[random_indices]
        X_remaining = np.delete(X, random_indices, axis=0)
        Y = np.random.uniform(low=low, high=high, size=(m,d))
        # Compute u_distances
        neigh_u = NearestNeighbors(n_neighbors=1, metric=metric).fit(X)
        u_distances, _ = neigh_u.kneighbors(Y)
        # Compute w_distances
        neigh_y = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_remaining)
        w_distances, _ = neigh_y.kneighbors(X_tilt)

    H = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))

    return H

def compute_clusters_distribution(cluster_labels):

    """
    :param cluster_labels: 1D array of cluster cluster_labels.
    :return: Standard Deviation that represent distribution across all clusters.
    """

    counts = Counter(cluster_labels)
    patient_counts = list(counts.values())
    std_dist = np.std(patient_counts)
    print("Patient count: ", patient_counts)
    print("STD:", std_dist)
    
    return std_dist