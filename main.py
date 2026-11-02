import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.utils import load_data, fill_missing, discretize
from src.id3 import ID3
from src.c45 import C45
from src.cart import CART

NUMERIC = ["income", "credit_score", "age"]


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    results = {}

    # Versions of the data with missing values filled
    Xtr, Xte = fill_missing(X_train, X_test)

    # ---- ID3: needs categorical data only ----
    Xtr_d, Xte_d = discretize(Xtr, Xte, NUMERIC)
    id3 = ID3(max_depth=5).fit(Xtr_d, y_train)
    results["ID3"] = accuracy(y_test, id3.predict(Xte_d))

    # ---- C4.5: handles numeric + missing values itself ----
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    c45 = C45(max_depth=5).fit(X_fit, y_fit)
    results["C4.5 (unpruned)"] = accuracy(y_test, c45.predict(X_test))
    c45.prune(X_val, y_val)
    results["C4.5 (pruned)"] = accuracy(y_test, c45.predict(X_test))

    # ---- CART ----
    cart = CART(max_depth=5, min_samples_leaf=5).fit(Xtr, y_train)
    results["CART"] = accuracy(y_test, cart.predict(Xte))

    # ---- scikit-learn baseline ----
    both = pd.get_dummies(pd.concat([Xtr, Xte]))
    A, B = both.iloc[:len(Xtr)], both.iloc[len(Xtr):]
    sk = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42).fit(A, y_train)
    results["sklearn"] = accuracy(y_test, sk.predict(B))

    # ---- Print results ----
    print("\nTest accuracy")
    print("-" * 30)
    for name, acc in results.items():
        print(f"{name:<18} {acc:.3f}")

    print("\nCART tree (top levels):")
    cart.print_tree()

    # ---- Save a bar chart ----
    plt.figure(figsize=(8, 4))
    plt.bar(results.keys(), results.values())
    plt.ylim(0.5, 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Decision tree algorithms on Loan Approval data")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved chart to results.png")


if __name__ == "__main__":
    main()