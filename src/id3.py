import numpy as np
import pandas as pd
from .utils import information_gain, majority_class


class Node:
    def __init__(self, prediction):
        self.prediction = prediction   # majority class at this node
        self.feature = None            # None means this node is a leaf
        self.children = {}             # feature value -> child node


class ID3:
    """Categorical features only. Split = highest information gain. Multiway splits."""

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        self.root = self._build(X, y, list(X.columns), 0)
        return self

    def _build(self, X, y, features, depth):
        node = Node(majority_class(y))
        if len(np.unique(y)) == 1 or not features:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        best_feature, best_gain = None, 0.0
        for f in features:
            groups = [y[(X[f] == v).to_numpy(dtype=bool)] for v in X[f].unique()]
            gain = information_gain(y, groups)
            if gain > best_gain:
                best_feature, best_gain = f, gain
        if best_feature is None:
            return node

        node.feature = best_feature
        remaining = [f for f in features if f != best_feature]
        for v in X[best_feature].unique():
            mask = (X[best_feature] == v).to_numpy(dtype=bool)
            node.children[v] = self._build(X[mask].reset_index(drop=True), y[mask], remaining, depth + 1)
        return node

    def _predict_row(self, row):
        node = self.root
        while node.feature is not None:
            value = row[node.feature]
            if value not in node.children:   # unseen value -> use majority here
                break
            node = node.children[value]
        return node.prediction

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._predict_row(r) for _, r in X.iterrows()])

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.feature is None:
            print(f"{indent}-> {node.prediction}")
            return
        for value, child in node.children.items():
            print(f"{indent}[{node.feature} = {value}]")
            self.print_tree(child, indent + "    ")