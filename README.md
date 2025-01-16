<p align="center">
  <img width="160" src="https://github.com/donlapark/coverforest/raw/main/doc/images/coverforest_96.png">
</p>

## 🌳 coverforest - Random Forest with Conformal Predictions

A fast and efficient implementation of conformal random forests for both classification and regression tasks. coverforest extends scikit-learn's random forest implementation to provide prediction sets/intervals with guaranteed coverage properties using conformal prediction methods.

coverforest provides three conformal prediction methods for random forests:
- CV+ (Cross-Validation+)[^1]
- Jackknife+-after-Bootstrap[^2]
- Split Conformal[^3]

The library provides two main classes: `CoverForestClassifier` and `CoverForestRegressor`, which follow the standard scikit-learn convention.
```python
from coverforest import CoverForestClassifier

clf = CoverForestClassifier(n_estimators=100, method='cv')
clf.fit(X_train, y_train)
y_pred, y_sets = clf.predict(X_test, alpha=0.05)  # 95% coverage sets
```

```python
from coverforest import CoverForestRegressor

reg = CoverForestRegressor(n_estimators=100, method='bootstrap')
reg.fit(X_train, y_train)
y_pred, y_intervals = reg.predict(X_test, alpha=0.05)  # 95% coverage intervals
```

## 🔧 Requirements

- Python >=3.9
- Scikit-learn >=1.6.0

## ⚡ Installation

You can install coverforest using pip:

```bash
pip install coverforest
```

Or install from source:

```bash
git clone https://github.com/donlapark/coverforest.git
cd coverforest
pip install .
```

## ✨ Features

- Fast implementation leveraging efficient random forest algorithms
- Support for three conformal prediction methods: CV+, Jackknife+-after-Bootstrap, and Split Conformal
- Automatic parameter tuning for optimal prediction set sizes in classification
- Parallel processing support for both training and prediction
- Seamless integration with scikit-learn's API
- Comprehensive uncertainty quantification through prediction sets/intervals
- Support for sample weights and custom cross-validation schemes

## 📖 References

1. Angelopoulos, A. N., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
2. Kim, B., Xu, C., & Barber, R. F. (2020). "Predictive inference with the jackknife+"
3. Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning in a Random World"

## 📜 License

MIT License

## Citation

If you use coverforest in your research, please cite:

```bibtex
@software{coverforest2024,
  author = {Your Name},
  title = {coverforest: Fast Conformal Random Forests},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/donlapark/coverforest}
}
```
