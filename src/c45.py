import numpy as np
import pandas as pd
from .utils import information_gain, gain_ratio, majority_class


class Node:
    def __init__(self, prediction):
        self.prediction = prediction
        self.feature = None        # None means leaf
        self.threshold = None      # None = categorical split, number = numeric split
        self.children = {}         # key -> child ("<=" / ">" for numeric)
        self.default_branch = None # where missing / unseen values go


class C45:
    """Gain ratio, numeric thresholds, missing values, and reduced-error pruning."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.root = self._build(X, y, list(X.columns), 0)
        return self

    # ---------- building ----------
    def _build(self, X, y, features, depth):
        node = Node(majority_class(y))
        if (len(np.unique(y)) == 1 or len(y) < self.min_samples_split or not features
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        best = self._best_split(X, y, features)
        if best is None:
            return node
        feature, threshold = best
        node.feature, node.threshold = feature, threshold

        values = X[feature].to_numpy()
        known = X[feature].notna().to_numpy()
        keys = np.empty(len(y), dtype=object)
        if threshold is None:
            keys[known] = values[known]
        else:
            keys[known] = np.where(values[known].astype(float) <= threshold, "<=", ">")
        # rows with a missing value follow the biggest branch
        node.default_branch = pd.Series(keys[known]).value_counts().idxmax()
        keys[~known] = node.default_branch

        remaining = features if threshold is not None else [f for f in features if f != feature]
        for key in set(keys):
            mask = keys == key
            node.children[key] = self._build(X[mask].reset_index(drop=True), y[mask], remaining, depth + 1)
        return node

    def _best_split(self, X, y, features):
        candidates = []  # (feature, threshold, gain, ratio)
        for f in features:
            known = X[f].notna().to_numpy()
            if known.sum() < 2:
                continue
            frac = known.mean()              # penalise features with missing values
            yk = y[known]
            vals = X[f].to_numpy()[known]
            if f in self.numeric_cols:
                res = self._best_threshold(vals.astype(float), yk)
                if res is None:
                    continue
                thr, gain, ratio = res
            else:
                thr = None
                groups = [yk[vals == v] for v in np.unique(vals)]
                gain, ratio = information_gain(yk, groups), gain_ratio(yk, groups)
            candidates.append((f, thr, gain * frac, ratio * frac))

        candidates = [c for c in candidates if c[2] > 1e-12]
        if not candidates:
            return None
        # C4.5 rule: only consider features with at least average gain, then pick best gain RATIO
        avg_gain = np.mean([c[2] for c in candidates])
        good = [c for c in candidates if c[2] >= avg_gain - 1e-12]
        best = max(good, key=lambda c: c[3])
        return best[0], best[1]

    def _best_threshold(self, vals, y):
        order = np.argsort(vals)
        v, l = vals[order], y[order]
        best, best_gain = None, 0.0
        for i in range(len(v) - 1):
            if v[i] == v[i + 1]:
                continue
            groups = [l[:i + 1], l[i + 1:]]
            g = information_gain(l, groups)
            if g > best_gain:
                best_gain = g
                best = ((v[i] + v[i + 1]) / 2, g, gain_ratio(l, groups))
        return best

    # ---------- predicting ----------
    def _route(self, node, value):
        if pd.isna(value):
            return node.default_branch
        if node.threshold is not None:
            return "<=" if value <= node.threshold else ">"
        return value if value in node.children else node.default_branch

    def _predict_row(self, node, row):
        while node.feature is not None:
            node = node.children[self._route(node, row[node.feature])]
        return node.prediction

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._predict_row(self.root, r) for _, r in X.iterrows()])

    # ---------- pruning (reduced-error pruning on a validation set) ----------
    def prune(self, X_val, y_val):
        self._prune(self.root, pd.DataFrame(X_val).reset_index(drop=True), np.asarray(y_val))
        return self

    def _prune(self, node, X, y):
        if node.feature is None:
            return
        keys = np.array([self._route(node, v) for v in X[node.feature]], dtype=object)
        for key, child in node.children.items():
            mask = keys == key
            self._prune(child, X[mask], y[mask])
        subtree_errors = sum(self._predict_row(node, r) != t for (_, r), t in zip(X.iterrows(), y))
        leaf_errors = int(np.sum(y != node.prediction))
        if leaf_errors <= subtree_errors:     # a leaf is no worse -> collapse the subtree
            node.feature, node.children = None, {}

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.feature is None:
            print(f"{indent}-> {node.prediction}")
            return
        for key, child in node.children.items():
            if node.threshold is None:
                print(f"{indent}[{node.feature} = {key}]")
            else:
                print(f"{indent}[{node.feature} {key} {node.threshold:.1f}]")
            self.print_tree(child, indent + "    ")