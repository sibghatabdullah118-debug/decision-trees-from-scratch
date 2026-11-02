# \# Decision Trees From Scratch: ID3, C4.5, CART

# 

# Implementations of three classic decision tree algorithms in Python/NumPy,

# compared on a synthetic Loan Approval dataset and against scikit-learn.

# 

# \## Algorithms

# 

# | Algorithm | Criterion | Feature types | Splits | Notes |

# |---|---|---|---|---|

# | ID3 | Information gain | Categorical only | Multiway | Numbers are binned into low/medium/high |

# | C4.5 | Gain ratio | Categorical + numeric | Multiway / threshold | Handles missing values, reduced-error pruning |

# | CART | Gini impurity | Categorical + numeric | Always binary | Depth and leaf-size limits |

# 

# \## Project structure

# 

# ```

# data/generate\_data.py   creates the synthetic dataset

# src/utils.py            entropy, gini, gain ratio, preprocessing

# src/id3.py  src/c45.py  src/cart.py

# main.py                 trains and compares all models

# tests/test\_trees.py     unit tests

# ```

# 

# \## How to run

# 

# ```bash

# python -m venv venv

# venv\\Scripts\\activate

# python -m pip install -r requirements.txt

# python data/generate\_data.py

# python main.py

# python -m pytest -v

# ```

# 

# \## Results

# 

# (PASTE YOUR ACCURACY TABLE HERE)

# 

# !\[results](results.png)

# 

# \## What I learned

# 

# (WRITE 3-4 SENTENCES IN YOUR OWN WORDS)

# 

# \## License

# MIT

