#!/usr/bin/env python3
"""
Module for Isolation Random Tree for outlier detection
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Isolation_Random_Tree(Decision_Tree):
    """
    Class that defines an isolation random tree, inheriting from Decision_Tree
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

    def np_extrema(self, arr):
        """
        Returns the min and max of an array
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Returns a random split criterion using Decision_Tree's base logic
        """
        return Decision_Tree.random_split_criterion(self, node)

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child node with depth value
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

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

        # Əgər data eyni nöqtələrdən ibarətdirsə və bölünmürsə, sonsuz dövrəni kəsirik
        if (np.sum(left_population) == 0 or
                np.sum(right_population) == 0):
            node.left_child = self.get_leaf_child(node, node.sub_population)
            node.right_child = self.get_leaf_child(node, node.sub_population)
            return

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
