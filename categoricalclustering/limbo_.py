
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

    
"""
import numpy as np
import pandas as pd
from information_bottleneck.information_bottleneck_algorithms.aIB_class import aIB


class Limbo:
    def __init__(self, b, max_nodes):
        self.n = None
        self.dcf_df = None
        self.m = None
        self.max_tree_nodes = max_nodes
        self.used_tree_nodes = 0
        self.tree_root = None

        self.tree_b = b

    class LimboTreeNode:
        def __init__(self, dcf, subtree=None):
            self.dcf = dcf
            self.subtree = subtree

    class LimboTree:
        def __init__(self, limbo_instance):
            self.limbo_instance = limbo_instance

            self.nodes = list()
            self.isLeaf = True

        def get_structure(self, level=0):
            s = f"{level*' '}{len(self.nodes)} nodes: "
            if not self.isLeaf:
                for node in self.nodes:
                    s += "\n"
                    s += node.subtree.get_structure(level+1)
            else:
                s += "leaf"
            return s

        def get_size(self):
            n = len(self.nodes)
            if not self.isLeaf:
                for node in self.nodes:
                    n+= node.subtree.get_size()
            return n
        def insert_node(self, node):
            if len(self.nodes) < self.limbo_instance.b:
                self.nodes.append(node)

        def get_leafs(self):
            if self.isLeaf:
                return self.nodes
            else:
                l = []
                for node in self.nodes:
                    l += node.subtree.get_leafs()
                return l

        def insert (self, node):
            if self.isLeaf:
                # leaf has free node: insert x there
                if len(self.nodes) < self.limbo_instance.tree_b:
                    self.nodes.append(node)
                    self.limbo_instance.used_tree_nodes += 1
                # leaf has no free node, but there is space: split leaf into two and return to parent
                elif self.limbo_instance.used_tree_nodes < self.limbo_instance.max_tree_nodes:

                    node_0, node_1 = Limbo.LimboTree.split_insert_trees(self, node)

                    if self == self.limbo_instance.tree_root:
                        self.limbo_instance.tree_root = Limbo.LimboTree(self.limbo_instance)
                        self.limbo_instance.tree_root.nodes = [node_0, node_1]
                        self.limbo_instance.tree_root.isLeaf = False
                        self.limbo_instance.used_tree_nodes = self.limbo_instance.tree_root.get_size()

                    else:
                        return node_0, node_1


                # otherwise, merge nodes
                else:
                    closest_cluster_to_input_idx, closets_cluster_to_input_distance = Limbo.LimboTree.find_closest_cluster_idx(self.nodes, node)
                    min_cluster_distance_idx,  min_cluster_distance = Limbo.LimboTree.find_closest_clusters(self.nodes)

                    # if input is closer to a cluster: merge input into cluster
                    if closets_cluster_to_input_distance < min_cluster_distance:
                        self.nodes[closest_cluster_to_input_idx].dcf = Limbo.merge_cluster_dcf(self.nodes[closest_cluster_to_input_idx].dcf, node.dcf)
                    # if two clusters are closer: merge clusters, input becomes new cluster
                    else:
                        self.nodes[min_cluster_distance_idx[0]].dcf = Limbo.merge_cluster_dcf(self.nodes[min_cluster_distance_idx[0]].dcf, self.nodes[min_cluster_distance_idx[1]].dcf)
                        self.nodes[min_cluster_distance_idx[1]] = node

            else:
                closest_cluster_idx, _ = Limbo.LimboTree.find_closest_cluster_idx(self.nodes, node)

                split_return = self.nodes[closest_cluster_idx].subtree.insert(node)

                # child was split into two subtrees, insert them at local height
                if split_return is not None:
                    node_0, node_1 = split_return

                    # update reference to node which was split
                    self.nodes[closest_cluster_idx] = node_0

                    #print()
                    #print(f"splitted {closest_cluster_idx}")
                    #print(self.get_structure())
                    #print(node_1.subtree.get_structure())
                    #print()

                    # if there is space: add second node
                    if len(self.nodes) < self.limbo_instance.tree_b:
                        self.nodes.append(node_1)
                        self.limbo_instance.used_tree_nodes = self.limbo_instance.tree_root.get_size()
                    else:
                        # perform split on current level
                        node_0, node_1 = Limbo.LimboTree.split_insert_trees(self, node_1)

                        # if current level is root, create new root
                        if self == self.limbo_instance.tree_root:
                            self.limbo_instance.tree_root = Limbo.LimboTree(self.limbo_instance)
                            self.limbo_instance.tree_root.nodes = [node_0, node_1]
                            self.limbo_instance.tree_root.isLeaf = False
                            self.limbo_instance.used_tree_nodes = self.limbo_instance.tree_root.get_size()
                        # no space on current level, but not root either => pass to calling tree node
                        else:
                            return  node_0, node_1

        """
        Find the closest cluster to a given node
        """
        @staticmethod
        def find_closest_cluster_idx(nodes, compare_node):
            # find min cluster
            closest_cluster_to_input_idx = -1
            closets_cluster_to_input_distance = float('inf')
            for dcf_idx, cluster_node in enumerate(nodes):
                distance = Limbo.delta_information(compare_node.dcf, cluster_node.dcf)
                if distance < closets_cluster_to_input_distance:
                    closets_cluster_to_input_distance = distance
                    closest_cluster_to_input_idx = dcf_idx

            return closest_cluster_to_input_idx, closets_cluster_to_input_distance

        """
        Find the closest clusters from a list of nodes
        """
        @staticmethod
        def find_closest_clusters(nodes):
            min_cluster_distance = float('inf')
            min_cluster_distance_idx = (-1, -1)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    distance = Limbo.delta_information(nodes[i].dcf, nodes[j].dcf)

                    if distance < min_cluster_distance:
                        min_cluster_distance = distance
                        min_cluster_distance_idx = (i, j)
            return min_cluster_distance_idx, min_cluster_distance

        @staticmethod
        def find_furthest_clusters(nodes):
            max_cluster_distance = 0
            max_cluster_distance_idx = (-1, -1)
            for i in range(len(nodes) - 1):
                for j in range(i + 1, len(nodes)):
                    distance = Limbo.delta_information(nodes[i].dcf, nodes[j].dcf)

                    if distance > max_cluster_distance:
                        max_cluster_distance = distance
                        max_cluster_distance_idx = (i, j)
            return max_cluster_distance_idx

        """
        Inserts a node (which can Link to a subtree) into a full tree by creating two nodes splitting the nodes.  
        """
        @staticmethod
        def split_insert_trees(tree, node):
            combined_nodes = tree.nodes.copy()
            combined_nodes.append(node)

            max_cluster_distance_idx = Limbo.LimboTree.find_furthest_clusters(combined_nodes)

            # insert input + old dcfs into new subtrees

            node_1 = combined_nodes.pop(max_cluster_distance_idx[1]) # order is important because j > i => remove j first !
            node_0 = combined_nodes.pop(max_cluster_distance_idx[0])

            node_0.subtree = Limbo.LimboTree(tree.limbo_instance)
            node_1.subtree = Limbo.LimboTree(tree.limbo_instance)



            for node in combined_nodes:
                d_first = Limbo.delta_information(node.dcf, node_0.dcf)
                d_second = Limbo.delta_information(node.dcf, node_1.dcf)

                if d_first < d_second:
                    node_0.subtree.nodes.append(node)
                    if node.subtree is not None:
                        node_0.subtree.isLeaf = False
                else:
                    node_1.subtree.nodes.append(node)
                    if node.subtree is not None:
                        node_1.subtree.isLeaf = False

            return node_0, node_1


    @staticmethod
    def delta_information(dcf_i, dcf_j):
        p_i, p_i_A = dcf_i.iloc[0], dcf_i.iloc[1:]  # p_i := p(c_i), p_i_A _= p(A | c_i)
        p_j, p_j_A = dcf_j.iloc[0], dcf_j.iloc[1:]  # NOTE: In the paper p_i is a shorthand for p(A|c_i) !
        p_combined = p_i + p_j
        p_bar = p_i / p_combined * p_i_A + p_j / p_combined * p_j_A
        jensen_shannon_divergence = p_i / p_combined * Limbo.KL(p_i_A, p_bar) + p_j / p_combined * Limbo.KL(p_j_A, p_bar)
        return (p_i + p_j) * jensen_shannon_divergence

    @staticmethod
    def delta_information_np(dcf_i, dcf_j):
        p_i, p_i_A = dcf_i[0], dcf_i[1:]  # p_i := p(c_i), p_i_A _= p(A | c_i)
        p_j, p_j_A = dcf_j[0], dcf_j[1:]  # NOTE: In the paper p_i is a shorthand for p(A|c_i) !
        p_combined = p_i + p_j
        p_bar = p_i / p_combined * p_i_A + p_j / p_combined * p_j_A
        jensen_shannon_divergence = p_i / p_combined * Limbo.KL(p_i_A, p_bar) + p_j / p_combined * Limbo.KL(p_j_A,
                                                                                                            p_bar)
        return (p_i + p_j) * jensen_shannon_divergence

    @staticmethod
    def find_closest_dfc_idx(dcfs, compare_dcf):
        # find min cluster
        closest_cluster_to_input_idx = -1
        closets_cluster_to_input_distance = float('inf')
        for dcf_idx, cluster_dcf in dcfs.iterrows():
            distance = Limbo.delta_information(compare_dcf, cluster_dcf)
            if distance < closets_cluster_to_input_distance:
                closets_cluster_to_input_distance = distance
                closest_cluster_to_input_idx = dcf_idx

        return closest_cluster_to_input_idx

    @staticmethod
    def KL(p, q):
        ratio = p / q
        ratio.fillna(1.0, inplace=True)
        log_p_q = np.zeros_like(ratio)
        log_p_q[ratio != 0] = np.log2(ratio[ratio != 0])
        return np.sum(p * log_p_q)

    @staticmethod
    def merge_cluster_dcf(dcf_i, dcf_j):
        result = dcf_i.copy()
        p_i, p_i_A = dcf_i.iloc[0], dcf_i.iloc[1:]  # p_i := p(c_i), p_i_A _= p(A | c_i)
        p_j, p_j_A = dcf_j.iloc[0], dcf_j.iloc[1:]  # NOTE: In the paper p_i is a shorthand for p(A|c_i) !

        p_sum = p_i + p_j
        result.iloc[0] = p_sum
        result.iloc[1:] = p_i / p_sum * p_i_A + p_j / p_sum * p_j_A
        return result


    def fit(self, data:pd.DataFrame, k:int):
        """

        :param data: One hot encoded dataframe
        :param k: Number of clusters
        :return:
        """
        self.dcf_df = data.copy()

        # find number of attributes, row sum for one hot encoded
        row_sum = data.sum(axis=1)
        self.m = row_sum.iloc[0]
        #assert np.all(row_sum == self.m)

        self.n = data.shape[0]
        self.dcf_df *= 1 / self.m

        self.dcf_df.insert(loc=0, column="p(x)", value=1/self.n)
        self.tree_root = self.LimboTree(self)

        # Phase 1: build tree to find cluster elements
        for idx, dcf in self.dcf_df.iterrows():
            self.tree_root.insert(Limbo.LimboTreeNode(dcf))

        leafs = self.tree_root.get_leafs()
        dcfs = pd.DataFrame([n.dcf for n in leafs])

        # Phase 2: AIB Clustering
        ib_instance = aIB(dcfs.values, k)
        ib_instance.run_IB_algo()
        p_t_given_y, p_x_given_t, p_t = ib_instance.get_results() # p_x_given_t are the cluster representatives

        cluster_centers = pd.DataFrame(p_x_given_t, columns=self.dcf_df.columns)

        # Phase 3: associate elements with cluster centers
        clustering = pd.DataFrame(index=self.dcf_df.index)
        clustering.insert(loc=0, column="cluster", value=-1)

        for i, dcf in self.dcf_df.iterrows():
            clustering.at[i, "cluster"] = Limbo.find_closest_dfc_idx(cluster_centers, dcf)

        return clustering, cluster_centers