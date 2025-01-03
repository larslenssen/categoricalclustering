"""Clustering for Categorical Data with Relative Entropy Dissimilarity in Python.

This package implements catRED:
Clustering for Categorical Data with Relative Entropy Dissimilarity.

Evaluation measures:

- Average Relative Entropy Score (ARES)
- Minimum Relative Entropy Contrast (MREC)

References:

| Lars Lenssen, Philip Stahmann, Christian Janiesch, Erich Schubert:
| Archetype Discovery from Taxonomies:
| A Method to Cluster Small Datasets of Categorical Data
| 58th Hawaii International Conference on System Sciences, HICSS 2025
| https://hdl.handle.net/10125/108984

Related Measures:

- Category Utility (CU)

Reference:

| M. A. Gluck and J. E. Corter:
| Information Uncertainty and the Utility of Categories
| Proc. Conf. of Cognitive Science Society. 1985, pp. 283–287

Related Algorithms:
- COOLCAT
- Limbo

References:

| Daniel Barbara, Yi Li and Julia Couto:
| COOLCAT: an entropy-based algorithm for categorical clustering
| Proceedings of the 2002 ACM CIKM
| DOI: 10.1145/584792.584888

| Periklis Andritsos, Panayiotis Tsaparas, Rene J. Miller and Kenneth C. Sevcik:
| LIMBO: Scalable Clustering of Categorical Data
| Advances in Database Technology - {EDBT} 2004
| DOI: 10.1007/978-3-540-24741-8\_9



"""
from . import catred_
from . import coolcat_
from . import limbo_
import pandas as pd
import numpy as np


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
    return catred_.catred(df, weights)

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
    return catred_.mrec(df, weights)

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
    return catred_.ares(df, weights)

#@jit(nopython=True)
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
    return catred_.category_utility(df)

def coolcat(df, k, num_batches=10, refit_proportion=0.4):
    """Coolcat

	References:

    | Daniel Barbara, Yi Li and Julia Couto:
    | COOLCAT: an entropy-based algorithm for categorical clustering
    | Proceedings of the 2002 ACM CIKM
    | DOI: 10.1145/584792.584888

	:param df: import data with column 'cluster' for the assignment
	:type df: pandas.DataFrame
    :param k: number of clusters
	:type k: integer

	:return: clustering result
	:rtype: pandas.DataFrame
	"""
    cc = coolcat_.CoolCat(k)
    result = cc.cluster(df, num_batches, refit_proportion)

    return catred_.catclustResult(None, None, None, None, result)

def limbo(df, k, tree_b=5, max_nodes=30):
    """Limbo

	References:

    | Periklis Andritsos, Panayiotis Tsaparas, Rene J. Miller and Kenneth C. Sevcik:
    | LIMBO: Scalable Clustering of Categorical Data
    | Advances in Database Technology - {EDBT} 2004
    | DOI: 10.1007/978-3-540-24741-8\_9

    Information Bottleneck Algorithm: 

    | M.Stark, J.Lewandowsky:
    | Information Bottleneck Algorithms in Python
    | https://goo.gl/QjBTZf

	:param df: import data with column 'cluster' for the assignment
	:type df: pandas.DataFrame
    :param k: number of clusters
	:type k: integer

	:return: clustering result
	:rtype: pandas.DataFrame
	"""
    alg = limbo_.Limbo(tree_b, max_nodes)
    result, center = alg.fit(df, k)
    return catred_.catclustResult(None, None, None, None, result)

def merge_onehot_categories(df):
    """
    Merges multiple adjacent columns that are one hot encoded into one column where the categories are index encoded.
    The new column name gives the categories split by '|' characters.
    """
    return catred_.merge_onehot_categories(df)


def analyse_linkagematrix(df, matrix, weights, number_of_cluster, title=None):
  catred_.analyse_linkagematrix(df, matrix, weights, number_of_cluster, title)

def analyse_clustering(df, title=None):
  # Form flat clusters from the hierarchical clustering defined by the linkage matrix Z
  catred_.analyse_clustering(df, title)