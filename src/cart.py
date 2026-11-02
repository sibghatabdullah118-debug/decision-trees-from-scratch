import numpy as np
import pandas as pd
from .utils import gini, majority_class


class Node:
    def __init__(self, prediction):
        self.prediction = prediction
        self.feature = None     # None means leaf
        self.threshold = None   # number (numeric) or category value (categorical)
        self.is_numeric = True
        self.left = None        # condition is True
        self.right = None       # condition is False


class CART:
    """Gini impurity, ALWAYS binary splits. Expects no missing values."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = Node(majority_class(y))
        if (len(np.unique(y)) == 1 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        feature, value, is_numeric, mask = self._best_split(X, y)
        if feature is None:
            return node

        node.feature, node.threshold, node.is_numeric = feature, value, is_numeric
        node.left = self._build(X[mask].reset_index(drop=True), y[mask], depth + 1)
        node.right = self._build(X[~mask].reset_index(drop=True), y[~mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n = len(y)
        best_score = gini(y)          # a split must beat the parent's impurity
        best = (None, None, None, None)
        for f in X.columns:
            col = X[f].to_numpy()
            is_numeric = f in self.numeric_cols
            if is_numeric:
                u = np.unique(col.astype(float))
                options = [(t, col.astype(float) <= t) for t in (u[:-1] + u[1:]) / 2]
            else:                      # binary split: "value" vs "everything else"
                options = [(v, col == v) for v in np.unique(col)]
            for value, mask in options:
                n_left = int(mask.sum())
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                score = n_left / n * gini(y[mask]) + n_right / n * gini(y[~mask])
                if score < best_score - 1e-12:
                    best_score, best = score, (f, value, is_numeric, mask)
        return best

    def _predict_row(self, row):
        node = self.root
        while node.feature is not None:
            v = row[node.feature]
            go_left = (v <= node.threshold) if node.is_numeric else (v == node.threshold)
            node = node.left if go_left else node.right
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
        cond = (f"{node.feature} <= {node.threshold:.1f}" if node.is_numeric
                else f"{node.feature} == {node.threshold}")
        print(f"{indent}if {cond}:")
        self.print_tree(node.left, indent + "    ")
        print(f"{indent}else:")
        self.print_tree(node.right, indent + "    ")