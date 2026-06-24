#!/usr/bin/env python3
"""
Module for Isolation Random Tree for outlier detection
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Class that defines an isolation random tree
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes the isolation random tree
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        String representation of the tree
        """
        return self.root.__str__()

    def depth(self):
        """
        Returns the maximum depth of the tree
        """
        return self.root.depth()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes in the tree
        """
        return self.root.count_nodes(only_leaves=only_leaves)

    def update_bounds(self):
        """
        Updates the bounds for each node
        """
        self.root.update_bounds()

    def get_leaves(self):
        """
        Returns the leaves of the tree
        """
        return self.root.get_leaves()

    def update_predict(self):
        """
        Updates the predict function for the isolation tree
        """
        def predict_node(node, x):
            if isinstance(node, Leaf):
                return node.value
            if x[node.feature] < node.threshold:
                return predict_node(node.left_child, x)
            else:
                return predict_node(node.right_child, x)

        self.predict = lambda X: np.array([predict_node(self.root, x) for x in X])

    def np_extrema(self, arr):
        """
        Returns the min and max of an array
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Returns a random split criterion
        """
        feature = self.rng.integers(0, self.explanatory.shape[1])
        feature_values = self.explanatory[node.sub_population, feature]
        
        if len(feature_values) == 0:
            return feature, 0.0
            
        min_val, max_val = self.np_extrema(feature_values)
        if min_val == max_val:
            return feature, min_val
            
        threshold = self.rng.uniform(min_val, max_val)
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child node
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates and returns an internal node child
        """
        node_child = Node()
        node_child.depth = node.depth + 1
        node_child.sub_population = sub_population
        return node_child

    def fit_node(self, node):
        """
        Recursively fits the node to split the data
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] < node.threshold
        )
        right_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] >= node.threshold
        )

        is_left_leaf = (
            node.depth + 1 >= self.max_depth or
            np.sum(left_population) <= self.min_pop
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            node.depth + 1 >= self.max_depth or
            np.sum(right_population) <= self.min_pop
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fits the isolation random tree to the explanatory data
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0],
            dtype='bool'
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
