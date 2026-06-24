#!/usr/bin/env python3
"""Module for building a decision tree"""

import numpy as np


class Node:
    """Represents a node in a decision tree"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Returns the maximum depth below this node"""
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """Returns the number of nodes below this node"""
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(
                only_leaves=only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(
                only_leaves=only_leaves)
        return count

    def left_child_add_prefix(self, text):
        """Add prefix for left child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns a string representation of the node"""
        if self.is_root:
            title = "root"
        else:
            title = "-> node"
        result = (f"{title} [feature={self.feature},"
                  f" threshold={self.threshold}]\n")
        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result.rstrip()

    def get_leaves_below(self):
        """Returns the list of all leaves below this node"""
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Recursively computes bounds for each node"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Computes the indicator function from lower and upper bounds"""
        def is_large_enough(x):
            """Returns True if all features > lower bounds"""
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower.keys()]),
                axis=0
            )

        def is_small_enough(x):
            """Returns True if all features <= upper bounds"""
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper.keys()]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0
        )


class Leaf(Node):
    """Represents a leaf node in a decision tree"""

    def __init__(self, value, depth=None):
        """Initialize a Leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of this leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 as a leaf is always counted"""
        return 1

    def __str__(self):
        """Returns a string representation of the leaf"""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Returns this leaf as a list"""
        return [self]

    def update_bounds_below(self):
        """Leaf has no children, nothing to update"""
        pass


class Decision_Tree():
    """Represents a decision tree"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns a string representation of the tree"""
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """Returns the list of all leaves in the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds for all nodes in the tree"""
        self.root.update_bounds_below()
