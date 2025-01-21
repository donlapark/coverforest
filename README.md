<p align="center">
  <img width="160" src="/doc/_static/img/coverforest_96.png">
</p>

## 🌳 coverforest - Random Forest with Conformal Predictions

A simple and fast implementation of conformal random forests for both classification and regression tasks. **coverforest** extends [scikit-learn](https://scikit-learn.org)'s random forest implementation to provide prediction sets/intervals with guaranteed coverage using conformal prediction methods.

**coverforest** provides three conformal prediction methods for random forests:
- CV+ (Cross-Validation+) [[1](#1), [2](#2)].
- Jackknife+-after-Bootstrap [[3]](#3).
- Split Conformal [[4]](#4).

The library provides two main classes: `CoverForestRegressor` for interval prediction and `CoverForestClassifier`. for set prediction.
Here are quick runs of the two classes:

```python
from coverforest import CoverForestRegressor

reg = CoverForestRegressor(n_estimators=100, method='bootstrap')  # using J+-a-Bootstrap
reg.fit(X_train, y_train)
y_pred, y_intervals = reg.predict(X_test, alpha=0.05)             # 95% coverage intervals
```

```python
from coverforest import CoverForestClassifier

clf = CoverForestClassifier(n_estimators=100, method='cv')  # using CV+
clf.fit(X_train, y_train)
y_pred, y_sets = clf.predict(X_test, alpha=0.05)            # 95% coverage sets
```

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

### Regularization in conformal set predictions

The classifier includes two regularization parameters $k$ and $\lambda$ that encourage smaller prediction sets [[5]](#5).

```python
clf = CoverForestClassifier(n_estimators=100, method='cv', k_init=2, lambda_init=0.1)
```

Automatic searching for suitable $k$ and $\lambda$ is also possible by specifying `k_init="auto"` and `lambda_init="auto"`, which are the default values of `CoverForestClassifier`.

### Performance Tips

Random forest leverages parallel computation by processing trees concurrently. Use the `n_jobs` parameter in `fit()` and `predict()` to control CPU usage (`n_jobs=-1` uses all cores).

For prediction, conformity score calculations require a memory array of size `(n_train × n_test × n_classes)`. To optimize performance with high `n_jobs` values, split large test sets into smaller batches.

See the documentation for more details and examples.

## 🔗 See Also

- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE): A Python package that provides scikit-learn-compatible wrappers for conformal classification and regression
- [conforest](https://github.com/knrumsey/conforest) An R implementation of random forest with inductive conformal prediction.
- [clover](https://github.com/Monoxido45/clover) A Python implementation of a regression forest method for conditional coverage ($`P(Y \vert X =x)`$) guarantee.
- [Conformal Prediction](https://github.com/aangelopoulos/conformal-prediction): Jupyter Notebook demonstrations of conformal prediction on various tasks, such as image classification, image segmentation, times series forecasting, and outlier detection
- [TorchCP](https://github.com/ml-stat-Sustech/TorchCP) A Python toolbox for Conformal Prediction in Deep Learning built on top of PyTorch
- [crepes](https://github.com/henrikbostrom/crepes) A Python package that implements standard and Mondrian conformal classifiers as well as standard, normalized and Mondrian conformal regressors and predictive systems.
- [nonconformist](https://github.com/donlnz/nonconformist): One of the first Python implementations of conformal prediction


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
@software{coverforest2025,
  author = {Donlapark Ponnoprat},
  title = {coverforest: Fast Conformal Random Forests},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/donlapark/coverforest}
}
```
