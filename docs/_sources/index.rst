###########
coverforest
###########

**coverforest** is a Python library that extends scikit-learn's random forest implementation to provide prediction sets/intervals with guaranteed coverage using conformal prediction methods. It offers a simple and efficient way to get uncertainty estimates for both classification and regression tasks.

**Useful links**:
`Source Repository <https://github.com/donlapark/coverforest>`__ |
`Issues & Ideas <https://github.com/donlapark/coverforest/issues>`__ |

Key Features
============
- Scikit-learn compatible API
- Three conformal prediction methods:
    - CV+ (Cross-Validation+) [1]_ [2]_
    - Jackknife+-after-Bootstrap [3]_
    - Split Conformal [4]_
- Efficient conformity score calculation with parallel processing support
- Regularized set predictions for classification tasks [5]_

Installation
============

You can install **coverforest** using pip:

.. code-block:: bash

    pip install coverforest

Requirements:

- Python >=3.9
- Scikit-learn >=1.6.0

Quick Start
===========

Classification Example
----------------------
.. code-block:: python

    from coverforest import CoverForestClassifier

    clf = CoverForestClassifier(n_estimators=100, method='cv')  # using CV+
    clf.fit(X_train, y_train)
    y_pred, y_sets = clf.predict(X_test, alpha=0.05)            # 95% coverage sets

Regression Example
------------------
.. code-block:: python

    from coverforest import CoverForestRegressor

    reg = CoverForestRegressor(n_estimators=100, method='bootstrap')  # using J+-a-Bootstrap
    reg.fit(X_train, y_train)
    y_pred, y_intervals = reg.predict(X_test, alpha=0.05)             # 95% coverage intervals


Performance Tips
================

- Use the ``n_jobs`` parameter in ``fit()`` and ``predict()`` to control parallel processing (``n_jobs=-1`` uses all CPU cores)
- For large test sets, consider batch processing to optimize memory usage when calculating conformity scores
- The memory requirement for prediction scales with ``(n_train × n_test × n_classes)``

References
==========

.. [1] Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with Valid and Adaptive Coverage. NeurIPS 2020.
.. [2] Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. Ann. Statist. 49(1), 486-507.
.. [3] Kim, B., Xu, C., & Barber, R. F. (2020). Predictive inference is free with the jackknife+-after-bootstrap. NeurIPS 2020.
.. [4] Vovk, V., Nouretdinov, I., Manokhin, V., & Gammerman, A. (2018). Cross-conformal predictive distributions. COPA 2018, 37-51.
.. [5] Angelopoulos, A. N., Bates, S., Jordan, M. I., & Malik, J. (2021). Uncertainty Sets for Image Classifiers using Conformal Prediction. ICLR 2021.

.. toctree::
    :maxdepth: 3
    :hidden:

    classification/index
    regression/index
    api
