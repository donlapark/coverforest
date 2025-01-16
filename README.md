<p align="center">
  <img width="160" src="https://github.com/donlapark/coverforest/raw/main/doc/images/coverforest_96.png">
</p>

## 🌳 coverforest - Random Forest with Conformal Predictions

A fast and simple implementation of conformal random forests for both classification and regression tasks. **coverforest** extends scikit-learn's random forest implementation to provide prediction sets/intervals with guaranteed coverage using conformal prediction methods.

**coverforest** provides three conformal prediction methods for random forests:
- CV+ (Cross-Validation+) [[1](#1), [2](#2)]
- Jackknife+-after-Bootstrap [[3]](#3)
- Split Conformal [[4]](#4)

The library provides two main classes: `CoverForestRegressor` and `CoverForestClassifier`.
Here are quick runs of the two classes:

```python
from coverforest import CoverForestRegressor

reg = CoverForestRegressor(n_estimators=100, method='bootstrap')
reg.fit(X_train, y_train)
y_pred, y_intervals = reg.predict(X_test, alpha=0.05)  # 95% coverage intervals
```

```python
from coverforest import CoverForestClassifier

clf = CoverForestClassifier(n_estimators=100, method='cv')
clf.fit(X_train, y_train)
y_pred, y_sets = clf.predict(X_test, alpha=0.05)  # 95% coverage sets
```

The classifier includes two regularization parameters $k$ and $\lambda$ that encourage smaller prediction sets [[5]](#5).

```python
clf = CoverForestClassifier(n_estimators=100, method='cv', k_init=2, lambda_init=0.1)
```

Automatic search for suitable $k$ and $\lambda$ is also possible by specifying `k_init="auto"` and `lambda_init="auto"`—These are the default values when instantiating `CoverForestClassifier` models.

See the documentation for more details and examples.

## ✨ Features

- Parallel processing support for both training and prediction
- Efficient conformity score calculations via Cython
- Automatic parameter tuning during the `fit()` call
- Seamless integration with scikit-learn's API

## 🔧 Requirements

- Python >=3.9
- Scikit-learn >=1.6.0

## ⚡ Installation

You can install **coverforest** using pip:

```bash
pip install coverforest
```

Or install from source:

```bash
git clone https://github.com/donlapark/coverforest.git
cd coverforest
pip install .
```

## 📖 References

<a id="1">[1]</a> Yaniv Romano, Matteo Sesia & Emmanuel J. Candès, "Classification with Valid and Adaptive Coverage", NeurIPS 2020.

<a id="2">[2]</a> Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas & Ryan J. Tibshirani, "Predictive inference with the jackknife+", Ann. Statist. 49 (1) 486-507, 2021.

<a id="3">[3]</a> Byol Kim, Chen Xu, Rina Foygel Barber, "Predictive inference is free with the jackknife+-after-bootstrap", NeurIPS 2020.

<a id="4">[4]</a> Vladimir Vovk, Ilia Nouretdinov, Valery Manokhin & Alexander Gammerman, "Cross-conformal predictive distributions", 37-51, COPA 2018.

<a id="5">[5]</a> Anastasios Nikolas Angelopoulos, Stephen Bates, Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image Classifiers using Conformal Prediction", ICLR 2021.

[6] Leo Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

## 📜 License

[BSD-3-Clause license](https://github.com/donlapark/coverforest/blob/main/LICENSE)

## 📝 Citation

If you use **coverforest** in your research, please cite:

```bibtex
@software{coverforest2024,
  author = {Your Name},
  title = {coverforest: Fast Conformal Random Forests},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/donlapark/coverforest}
}
```
