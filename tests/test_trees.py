import numpy as np
import pandas as pd
from src.utils import entropy, gini, information_gain
from src.id3 import ID3
from src.c45 import C45
from src.cart import CART

X = pd.DataFrame({
    "weather": ["sunny", "sunny", "rain", "rain", "overcast", "overcast"],
    "windy": ["no", "yes", "no", "yes", "no", "yes"],
})
y = np.array(["no", "no", "yes", "no", "yes", "yes"])


def test_entropy():
    assert entropy(np.array(["a", "a"])) == 0
    assert np.isclose(entropy(np.array(["a", "b"])), 1.0)


def test_gini():
    assert gini(np.array(["a", "a"])) == 0
    assert np.isclose(gini(np.array(["a", "b"])), 0.5)


def test_information_gain_perfect_split():
    yy = np.array(["y", "y", "n", "n"])
    assert np.isclose(information_gain(yy, [yy[:2], yy[2:]]), 1.0)


def test_id3_fits_toy_data():
    assert (ID3().fit(X, y).predict(X) == y).all()


def test_c45_fits_toy_data():
    assert (C45().fit(X, y).predict(X) == y).all()


def test_cart_fits_toy_data():
    assert (CART().fit(X, y).predict(X) == y).all()


def test_numeric_threshold():
    Xn = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12]})
    yn = np.array(["a", "a", "a", "b", "b", "b"])
    assert C45().fit(Xn, yn).root.threshold == 6.5
    assert CART().fit(Xn, yn).root.threshold == 6.5