import numpy as np
import pandas as pd


# ---------- Math used by the trees ----------
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))


def information_gain(y, groups):
    """Entropy before the split minus weighted entropy after it."""
    n = sum(len(g) for g in groups)
    after = sum(len(g) / n * entropy(g) for g in groups if len(g) > 0)
    return entropy(y) - after


def split_info(groups):
    n = sum(len(g) for g in groups)
    sizes = np.array([len(g) / n for g in groups if len(g) > 0])
    return float(-np.sum(sizes * np.log2(sizes)))


def gain_ratio(y, groups):
    si = split_info(groups)
    if si == 0:
        return 0.0
    return information_gain(y, groups) / si


def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


# ---------- Data helpers ----------
def load_data(path="data/loan_data.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns="approved")
    y = df["approved"].to_numpy()
    return X, y


def fill_missing(X_train, X_test):
    """Fill gaps using TRAIN statistics only (median for numbers, mode for text)."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for c in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            fill = X_train[c].median()
        else:
            fill = X_train[c].mode()[0]
        X_train[c] = X_train[c].fillna(fill)
        X_test[c] = X_test[c].fillna(fill)
    return X_train, X_test


def discretize(X_train, X_test, cols, labels=("low", "medium", "high")):
    """Turn numbers into low/medium/high so ID3 (categorical only) can use them."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for c in cols:
        edges = X_train[c].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        edges[0], edges[-1] = -np.inf, np.inf
        X_train[c] = pd.cut(X_train[c], bins=edges, labels=list(labels)).astype(str)
        X_test[c] = pd.cut(X_test[c], bins=edges, labels=list(labels)).astype(str)
    return X_train, X_test