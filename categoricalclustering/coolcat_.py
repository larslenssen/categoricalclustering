"""Coolcat

	References:

    | Daniel Barbara, Yi Li and Julia Couto:
    | COOLCAT: an entropy-based algorithm for categorical clustering
    | Proceedings of the 2002 ACM CIKM
    | DOI: 10.1145/584792.584888
    
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class CoolCat:
    def __init__(self, k:int, init_proportion=0.1):
        assert k>= 2, "Number of Clusters must be >=2"
        self.k = k
        self.init_proportion = init_proportion
        self.cluster_centers = []
        self.clustering = None

    def cluster(self, df, num_batches, refit_proportion=0.1, rng = np.random.default_rng()):
        
        df = df.copy().astype(np.float64)
        df.insert(loc=0, column="p(x)", value=1)

        self.clustering = pd.DataFrame(index=df.index)
        self.clustering.insert(loc=0, column="cluster", value=-1)
        self.clustering = self.clustering.astype({"cluster":'int32'})

        # initialize
        init_sample_size = int(self.init_proportion*len(df))
        assert init_sample_size >= self.k, f"The given proportion {self.init_proportion} results in {init_sample_size} datapoints for initialization, which is too small for {self.k} Clusters."

        sample_idx = rng.choice(len(df), init_sample_size)

        if len(self.cluster_centers) == 0:
            used_samples = self.initialize_clusters(df.iloc[sample_idx] )
            df.drop(index=used_samples,inplace=True)

        #for idx, p in tqdm(df.iterrows(), total=df.shape[0]):
        #    self.assign_point(p, idx)


        for batch_num, batch in df.groupby(np.arange(len(df))//num_batches):
            for idx, element in batch.iterrows():
                self.assign_point(element, idx)

                assert self.clustering.loc[idx, "cluster"] != -1, f"Element {idx} was not assigned"

            self.refit_batch(batch, refit_proportion)

        return self.clustering

    @staticmethod
    def expected_entropy(record_a: pd.Series, record_b:  pd.Series) -> np.double:
        p = (record_a.iloc[1:] + record_b.iloc[1:]) / 2
        log_p = np.zeros_like(p)

        # masking is neccessary due to log(0) being inf. Would be ignored by nansum, but still throws errors
        # for entropy, p log p = 0 for lim p = 0
        log_p[p > 0] = np.log2(p [p > 0])
        e = - np.nansum(p * log_p)
        return e

    def initialize_clusters(self, sample: pd.DataFrame):
        max_entropy = -1
        max_entropy_idx = -1, -1

        # find first two clusters
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # pairwise_entropy[i, j] = entropy(df[i], df[j])
                e = self.expected_entropy(sample.iloc[i], sample.iloc[j])
                if e > max_entropy:
                    max_entropy = e
                    max_entropy_idx = i, j

        self.cluster_centers = [sample.iloc[max_entropy_idx[0]], sample.iloc[max_entropy_idx[1]]]
        self.clustering.at[sample.iloc[max_entropy_idx[0]].name, "cluster"] = 0
        self.clustering.at[sample.iloc[max_entropy_idx[1]].name, "cluster"] = 1

        used_samples = []
        used_samples.append(sample.iloc[max_entropy_idx[0]].name)
        used_samples.append(sample.iloc[max_entropy_idx[1]].name)


        # find following initial cluster, which are idx = arg max_j min_c=[0,...,j-1] E( idx, C_c )
        for current_cluster_idx in range(2, self.k):
            max_entropy = 0
            max_entropy_idx = -1

            for i in range(len(sample)):
                if i in used_samples:
                    continue

                min_entropy = float('inf')
                for k in range(current_cluster_idx):
                    e = self.expected_entropy(self.cluster_centers[k], sample.iloc[i])
                    if e < min_entropy:
                        min_entropy = e
                if min_entropy > max_entropy:
                    max_entropy = min_entropy
                    max_entropy_idx = i

            self.cluster_centers.append(sample.iloc[max_entropy_idx])
            self.clustering.at[sample.iloc[max_entropy_idx].name, "cluster"] = current_cluster_idx

            used_samples.append(sample.iloc[max_entropy_idx].name)

        return used_samples

    def assign_point(self, p, p_idx):
        min_entropy = float('inf')
        min_entropy_cluster_idx = -1

        for i in range(self.k):
            e = self.expected_entropy(self.cluster_centers[i], p)
            if e < min_entropy:
                min_entropy = e
                min_entropy_cluster_idx = i

        self.cluster_centers[min_entropy_cluster_idx] = self.calculate_new_center(self.cluster_centers[min_entropy_cluster_idx], p)
        self.clustering.at[p_idx, "cluster"] = min_entropy_cluster_idx


    @staticmethod
    def calculate_new_center(old_center, new_point):
        n_old = old_center.iloc[0]
        n_new = new_point.iloc[0]  #should be 1 for new point
        n = n_old + n_new
        new_values = n_old / n * old_center.iloc[1:] + n_new / n  * new_point.iloc[1:]
        old_center.iloc[1:] = new_values
        old_center.iloc[0] = n
        return old_center

    @staticmethod
    def calculate_fitting_probability(cluster_center, element):
        attributes_probabilities = cluster_center.iloc[1:] - 1/cluster_center.iloc[0]
        # we ignore all occurrences caused by the element
        # we can subtract from all attributes, as not occurring elements get multiplied by 0 now
        p = np.sum(attributes_probabilities * element.iloc[1:])

        return p

    def refit_batch(self, batch, refit_proportion):
        m = int(len(batch.index) * refit_proportion)

        fit = np.empty(len(batch.index) )
        for i, (idx, row) in enumerate(batch.iterrows()):
            center = self.cluster_centers[self.clustering.loc[idx, "cluster"]]
            fit[i] = self.calculate_fitting_probability(center, row)

        refit = fit.argsort()[:m] # min_m elements

        for element_id in refit:
            element = batch.iloc[element_id]

            min_entropy = float('inf')
            min_entropy_cluster_idx = -1

            for i in range(self.k):
                e = self.expected_entropy(self.cluster_centers[i], element)
                if e < min_entropy:
                    min_entropy = e
                    min_entropy_cluster_idx = i

            current_cluster = self.clustering.loc[element.name, "cluster"]
            if min_entropy_cluster_idx != current_cluster:
                #remove element from old cluster center
                c = self.cluster_centers[current_cluster]
                num_c = c.iloc[0]
                element_contribution = 1/num_c * element.iloc[1:]
                p_att = c.iloc[1:] - element_contribution
                self.cluster_centers[current_cluster] = pd.concat([pd.Series(num_c-1), p_att])
                #set new cluster
                self.cluster_centers[min_entropy_cluster_idx] = self.calculate_new_center(self.cluster_centers[min_entropy_cluster_idx], element)
                self.clustering.at[element.name, "cluster"] = min_entropy_cluster_idx