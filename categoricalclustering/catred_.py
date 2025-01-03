"""Clustering for Categorical Data with Relative Entropy Dissimilarity in Python.

This package implements catRED:
Clustering for Categorical Data with Relative Entropy Dissimilarity.

Evaluation measures:

- Average Relative Entropy Score (ARES)
- Minimum Relative Entropy Contrast (MREC)
- Category Utility (CU)

References:

| Lars Lenssen, Philip Stahmann, Christian Janiesch, Erich Schubert:
| Archetype Discovery from Taxonomies:
| A Method to Cluster Small Datasets of Categorical Data
| 58th Hawaii International Conference on System Sciences, HICSS 2025
| https://hdl.handle.net/10125/108984
"""

import numpy as np
import pandas as pd
from numba import jit
from tqdm.auto import trange
from . import visualization 
from scipy.cluster.hierarchy import fcluster

class catclustResult:
	"""
	categorical clustering result

	:param linkage_matrix: Linkage Matrix of the resulting hierarchical clusering
	:type linkage_matrix: ndarray

	:param binary_maximum: Maximimum number of categories of one column
	:type binary_maximum: int

	:param c_probabilities_categorical: Probabilities of the different features
	:type c_probabilities_categorical: ndarray

	:param pnorm: pnorm
	:type pnorm: float

    :param clusters: cluster assignments, only for non hierarchical clustering
	:type data: pandas.DataFrame

	"""
	def __init__(self, linkage_matrix=None, binary_maximum=None, c_probabilities_categorical=None, pnorm=None, clusters=None):
		self.linkage_matrix = linkage_matrix
		self.binary_maximum = binary_maximum
		self.c_probabilities_categorical = c_probabilities_categorical
		self.pnorm = pnorm
		self.clusters = clusters

	def __repr__(self):
		return f"catclustResult(linkage_matrix={self.linkage_matrix}, binary_maximum={self.binary_maximum}, c_probabilities_categorical={self.c_probabilities_categorical}, pnorm={self.pnorm}, clusters={self.clusters})"
    

def catred(df, weights=None):
    """catRED clustering

	Clustering for Categorical Data with Relative Entropy Dissimilarity

	References:

    | Lars Lenssen, Philip Stahmann, Christian Janiesch, Erich Schubert:
    | Archetype Discovery from Taxonomies:
    | A Method to Cluster Small Datasets of Categorical Data
    | 58th Hawaii International Conference on System Sciences, HICSS 2025
    | https://hdl.handle.net/10125/108984

	:param df: import data
	:type df: pandas.DataFrame
	:param weights: weights of the different features
	:type weights: ndarray

	:return: categorical clustering result
	:rtype: catclustResult
	"""
    if weights is None:
        weights = pd.Series(np.ones(len(df.columns)), index=df.columns)

    weights = weights.values

    data = df.values

    #moved in from top level cell
    binary_maximum = np.amax(data)
    c_probabilities_categorical = calculate_probabilities_categorical(data, binary_maximum)


    pnorm = ((-c_probabilities_categorical * np.log2(c_probabilities_categorical,
                                                    where=c_probabilities_categorical > 0)).sum(axis=0) * np.array(weights)).sum()

    max_value = np.amax(data) if binary_maximum is None else binary_maximum
    # Create a 2D numpy array of zeros
    cluster_counts = np.zeros((2 * len(data) - 1, max_value + 1, data.shape[1] + 1), dtype=np.int64)
    # Set the column corresponding to the cluster of each point to 1
    for i in range(len(data)):
        cluster_counts[i] = np.hstack(
            (np.zeros((max_value + 1, 1), dtype=np.int64), calculate_counts_categorical(data[i], max_value)))
        cluster_counts[i, 0, 0] = 1
    distance_matrix = init_distance_matrix(cluster_counts, c_probabilities_categorical, pnorm, weights, binary_maximum)
    linkage_matrix = np.zeros((len(data) - 1, 4), dtype=np.float64)
    for c in trange(len(data) - 1, desc="Clustering"):
        new_index = len(data) + c
        # Find the two closest clusters
        i, j, distance = find_closest_clusters(distance_matrix, new_index)
        # Merge the two closest clusters
        cluster_counts[new_index] = cluster_counts[i] + cluster_counts[j]

        # Add the merge information to the linkage matrix
        linkage_matrix[c] = np.array([i, j, distance, cluster_counts[new_index, 0, 0]])
        # Set the count in the old clusters to zero
        cluster_counts[i, 0, 0] = 0
        cluster_counts[j, 0, 0] = 0
        distance_matrix[i, i] = -1
        distance_matrix[j, j] = -1
        c += 1
        update_distance_matrix(cluster_counts, distance_matrix, new_index, c_probabilities_categorical, pnorm, weights, binary_maximum)

    return catclustResult(linkage_matrix, binary_maximum, c_probabilities_categorical, pnorm)


#@jit(nopython=True)
def mrec(df, weights = None):
    """Minimum Relative Entropy Contrast (MREC)

	References:

    | Lars Lenssen, Philip Stahmann, Christian Janiesch, Erich Schubert:
    | Archetype Discovery from Taxonomies:
    | A Method to Cluster Small Datasets of Categorical Data
    | 58th Hawaii International Conference on System Sciences, HICSS 2025
    | https://hdl.handle.net/10125/108984

	:param data: import data with column 'cluster' for the assignment
	:type data: pandas.DataFrame
	:param weights: weights of the different features
	:type weights: ndarray

	:return: ARES
	:rtype: float
	"""
    data = df.values
    if weights is None:
        weights = pd.Series(np.ones(len(df.columns)), index=df.columns)

    weights = weights.values
    
        #moved in from top level cell
    binary_maximum = np.amax(data)
    c_probabilities_categorical = calculate_probabilities_categorical(data, binary_maximum)
    number_of_cluster = df['cluster'].nunique()
    
    # Create a 2D numpy array of zeros
    cluster_counts = np.zeros((number_of_cluster, binary_maximum+1, data.shape[1]), dtype=np.int64)
    cluster_index = data.shape[1] - 1
    # Set the column corresponding to the cluster of each point to 1
    for i in range(len(data)):
      cluster_counts[data[i,cluster_index]-1] += np.hstack((np.zeros((binary_maximum+1, 1), dtype=np.int64), calculate_counts_categorical(data[i,:-1], binary_maximum)))
      cluster_counts[data[i,cluster_index]-1, 0, 0] += 1
    # Initialize the cluster quality value
    mrec = float('inf')
    # calculate for each ccluster
    mrec_clust_best = float('-inf')
    for i in range(number_of_cluster):
        mrec_clust_own = res(cluster_counts[i], Y=None, p=c_probabilities_categorical, weights=weights)
        mrec_clust_best = float('-inf')
        for j in range(number_of_cluster):
          if i != j:
            mrec_clust_tmp = res(cluster_counts[i] + cluster_counts[j], Y=None, p=c_probabilities_categorical, weights=weights)
            if mrec_clust_tmp > mrec_clust_best: mrec_clust_best = mrec_clust_tmp
        mrec_clust_tmp = mrec_clust_own - mrec_clust_best
        if mrec_clust_tmp < mrec: mrec = mrec_clust_tmp
    return mrec

#@jit(nopython=True)
def ares(df, weights = None):
    """Average Relative Entropy Score (ARES)

	References:

    | Lars Lenssen, Philip Stahmann, Christian Janiesch, Erich Schubert:
    | Archetype Discovery from Taxonomies:
    | A Method to Cluster Small Datasets of Categorical Data
    | 58th Hawaii International Conference on System Sciences, HICSS 2025
    | https://hdl.handle.net/10125/108984

	:param data: import data with column 'cluster' for the assignment
	:type data: pandas.DataFrame
	:param weights: weights of the different features
	:type weights: ndarray

	:return: MREC
	:rtype: float
	"""
    data = df.values
    if weights is None:
        weights = pd.Series(np.ones(len(df.columns)), index=df.columns)

    weights = weights.values
	
	#moved in from top level cell
    binary_maximum = np.amax(data)
    c_probabilities_categorical = calculate_probabilities_categorical(data, binary_maximum)
    number_of_cluster = df['cluster'].nunique()

    # Create a 2D numpy array of zeros
    cluster_counts = np.zeros((number_of_cluster, binary_maximum+1, data.shape[1]))
    cluster_index = data.shape[1]-1
    # Set the column corresponding to the cluster of each point to 1
    for i in range(len(data)):
      cluster_counts[data[i,cluster_index]-1] += np.hstack((np.zeros((binary_maximum+1, 1)).astype(np.int64), calculate_counts_categorical(data[i,:-1], binary_maximum)))
      cluster_counts[data[i,cluster_index]-1, 0, 0] += 1
    # Initialize the cluster quality value
    ares = 0
    for i in range(number_of_cluster):
        ares += (cluster_counts[i, 0, 0] / len(data) )  * res(cluster_counts[i], Y=None, p=c_probabilities_categorical, weights=weights)
    ares = ares
    return ares


def category_utility(df):
    """Category Utility (CU)

	References:

    | M. A. Gluck and J. E. Corter:
    | Information Uncertainty and the Utility of Categories
    | Proc. Conf. of Cognitive Science Society. 1985, pp. 283–287

	:param data: import data with column 'cluster' for the assignment
	:type data: pandas.DataFrame

	:return: CU
	:rtype: float
	"""
    data = df.values

    binary_maximum = np.amax(data[:, :-1]) # exclude the cluster column from bm
    c_probabilities_categorical = calculate_probabilities_categorical(data[:, :-1], binary_maximum)
    number_of_cluster = df['cluster'].nunique()

    # Create a 2D numpy array of zeros
    cluster_counts = np.zeros((number_of_cluster, binary_maximum + 1, data.shape[1]))
    cluster_index = data.shape[1] - 1
    # Set the column corresponding to the cluster of each point to 1
    for i in range(len(data)):
        cluster_counts[data[i, cluster_index] - 1] += np.hstack((np.zeros((binary_maximum + 1, 1)).astype(np.int64), calculate_counts_categorical(data[i, :-1],  binary_maximum)))
        cluster_counts[data[i, cluster_index] - 1, 0, 0] += 1
    # Initialize the cluster quality value
    cu = 0
    for i in range(number_of_cluster):
        cu += (cluster_counts[i, 0, 0] / len(data)) * cu_single(cluster_counts[i], p=c_probabilities_categorical)
    cu = cu
    return cu


#@jit(nopython=True)
def cu_single(X, p):
    Xlen = X[0,0]
    d =  0
    for i in range(1, X.shape[1]):
        for j in range(len(X)):
            if p[j, i-1] == 0.0: continue # never in the data set
            p_i = X[j,i] / Xlen
            if p_i > 0:
              p_d = p[j, i-1]
              d +=  p_i**2 - p_d**2
    return d



def analyse_linkagematrix(df, matrix, weights, number_of_cluster, title=None):
  df_temp = df.copy()
  # Form flat clusters from the hierarchical clustering defined by the linkage matrix Z
  df_temp['cluster'] = fcluster(matrix, number_of_cluster, criterion='maxclust')
  num_clusters = len(df_temp['cluster'].unique())
  #moved in from top level cell
  prototypes = visualization.get_prototypes(df_temp.values, num_clusters)
  visualization.plot_dendogram(matrix, title + f'quality of the clustering (MREC): {round(mrec(df_temp, weights), 2)}', number_of_cluster)
  visualization.plot_clusters_plotly(df_temp, prototypes)

def analyse_clustering(df, title=None):
  # Form flat clusters from the hierarchical clustering defined by the linkage matrix Z
  #moved in from top level cell
  num_clusters = len(df['cluster'].unique())
  prototypes = visualization.get_prototypes(df.values, num_clusters)
  visualization.plot_clusters_plotly(df, prototypes, title + f'quality of the clustering: {round(mrec(df),2)}')


#@jit(nopython=True)
def calculate_probabilities_categorical(X, bin_max):
    return calculate_counts_categorical(X, bin_max) / len(X)


#@jit(nopython=True)
def calculate_counts_categorical(X, bin_max):
    #X = X.reshape(1, -1).astype(np.int64, copy=False) if X.ndim == 1 else X.astype(np.int64, copy=False)
    if X.ndim == 1: X = X.copy().reshape(1, -1)
    counts = np.zeros((bin_max + 1, X.shape[1]), dtype=np.int64)
    for i in range(X.shape[1]):
        counts[:np.bincount(X[:, i]).size, i] = np.bincount(X[:, i])
    return counts

#@jit(nopython=True)
def init_distance_matrix(cluster_counts, p, pnorm, weights, binary_maximum):
    n_lm = len(cluster_counts)
    n = int((n_lm + 1) / 2)
    #max_value = binary_maximum
    distance_matrix = np.zeros((n_lm, n_lm))
    for x in range(n):
      for y in range(n):
        distance_matrix[x, y] = red(cluster_counts[x], cluster_counts[y], p, pnorm, weights)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

#@jit(nopython=True)
def update_distance_matrix(cluster_counts, distance_matrix, i, p, pnorm, weights, binary_maximum):
    #max_value = binary_maximum
    for x in range(i+1):
      if cluster_counts[x, 0, 0] != 0 and x != i:
        distance_matrix[i, x] = red(cluster_counts[i], cluster_counts[x], p, pnorm, weights)
        distance_matrix[x, i] = distance_matrix[i, x]
    distance_matrix[i, i] = 0

#@jit(nopython=True)
def find_closest_clusters(distance_matrix, new_index):
    min_distance = np.inf
    closest_clusters = (-1, -1, np.inf)
    for i in range(new_index):
        if distance_matrix[i, i] != -1:
            for j in range(i + 1, new_index):
                if distance_matrix[j, j] != -1:
                    if distance_matrix[i, j] < min_distance:
                        min_distance = distance_matrix[i, j]
                        closest_clusters = (i, j, distance_matrix[i, j])
    return closest_clusters

def merge_onehot_categories(df):
    """
    Merges multiple adjacent columns that are one hot encoded into one column where the categories are index encoded.
    The new column name gives the categories split by '|' characters.
    """
    for i in range(0, len(df.columns) - 1):
        for j in range(i+1, len(df.columns) ):
            group_df = df.iloc[:, i: j + 1 ]
            # Check if a one appears two or more times in at least one row
            multiple_ones_appear = group_df.apply(lambda x: (x == 1).sum() >= 2, axis=1).any()
            # Count all ones in the group of columns
            count_ones = (group_df == 1).sum().sum()

            if multiple_ones_appear or count_ones > len(df):
                break
            elif count_ones == len(df):
                # Merge the group of columns from binary values to a string
                merged = group_df.apply(lambda x: ''.join(x.astype(str)), axis=1)

                # Convert the merged string to a categorical value
                merged = merged.astype('category')

                # Replace the group of columns with the merged column
                df[group_df.columns[0]] = merged.cat.codes
                # Create the new column name by concatenating the names of the merged columns
                new_column_name = '|'.join(group_df.columns[::-1])

                # Rename the replaced column
                df = df.rename(columns={group_df.columns[0]: new_column_name})

                df = df.drop(columns=group_df.columns[1:])

    return df

#@jit(nopython=True)
def red(X, Y, p, pnorm, weights):
  return np.power(2, -res(X,Y,p,weights) / pnorm)

#@jit(nopython=True)
def res(X, Y, p, weights):
    XYcounts = X
    XYlen = X[0,0]
    if Y is not None:
      XYcounts = X + Y
      XYlen = X[0,0] + Y[0,0]
    d =  0
    for i in range(1, XYcounts.shape[1]):
        for j in range(len(XYcounts)):
            if p[j, i-1] == 0.0: continue # never in the data set
            p_i = XYcounts[j,i] / XYlen
            if p_i > 0:
              p_d = p[j, i-1]
              d += weights[i-1] * p_i * np.log2(p_i  / p_d)
    return d

#@jit(nopython=True)
def res_dif(X, Y, p, weights):
  sim_union = res(X,Y,p,weights)
  #if Y is None: return np.power(2, -sim_union)
  sim_X = res(X,None,p,weights)
  sim_Y = res(Y,None,p,weights)
  return ((sim_X+sim_Y)/2) - sim_union


