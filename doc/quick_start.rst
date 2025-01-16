Quick Start Guide
=================

This guide will help you get started with **coverforest**, a Python package for conformal random forests.

Installation
------------

Install coverforest using pip:

.. code-block:: bash

    pip install coverforest

Basic Usage
-----------

Classification
^^^^^^^^^^^^^^

``CoverForestClassifier`` provides prediction sets that contain the true label with high probability:

.. code-block:: python

    from coverforest import CoverForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20,
                             n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit classifier
    clf = CoverForestClassifier(
        n_estimators=100,    # Number of forests
        method='cv',         # Use CV+ method
        cv=5,               # 5-fold cross-validation
        n_subestimators=50  # Trees per forest
    )

    # Fit with auto-tuning of k and lambda parameters
    clf.fit(X_train, y_train, alpha=0.1)  # 90% coverage target

    # Make predictions
    y_pred, y_sets = clf.predict(X_test, alpha=0.1)

    # y_pred contains point predictions
    # y_sets contains prediction sets (default: list of arrays)

    # For binary output format
    y_pred, y_sets = clf.predict(X_test, alpha=0.1, binary_output=True)
    # y_sets is now a binary array of shape (n_samples, n_classes)

Parameters k and λ
''''''''''''''''''

The classifier uses two regularization parameters:

- ``k``: Penalizes prediction sets containing more than k classes
- ``λ`` (lambda): Controls the strength of the k-penalty

You can set these manually:

.. code-block:: python

    clf = CoverForestClassifier(
        n_estimators=100,
        k_init=2,           # Maximum 2 classes per set
        lambda_init=0.1     # Moderate regularization
    )

Or use automatic tuning (default):

.. code-block:: python

    clf = CoverForestClassifier(
        n_estimators=100,
        k_init="auto",      # Auto-select k
        lambda_init="auto"  # Auto-select lambda
    )

Regression
^^^^^^^^^^

``CoverForestRegressor`` provides prediction intervals that contain the true value with high probability:

.. code-block:: python

    from coverforest import CoverForestRegressor
    from sklearn.datasets import make_regression

    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20,
                          n_informative=15, noise=0.1,
                          random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit regressor
    reg = CoverForestRegressor(
        n_estimators=100,
        method='bootstrap',  # Use Jackknife+-after-Bootstrap
        n_subestimators=50
    )

    # Fit the model
    reg.fit(X_train, y_train)

    # Make predictions with 95% coverage intervals
    y_pred, y_intervals = reg.predict(X_test, alpha=0.05)

    # y_pred contains point predictions
    # y_intervals contains (lower, upper) bounds for each prediction

Method Selection
----------------

coverforest supports three conformal prediction methods:

1. ``'cv'``: CV+ method (default)
    - Uses K-fold cross-validation
    - Good balance of computational cost and efficiency

2. ``'bootstrap'``: Jackknife+-after-Bootstrap
    - Uses bootstrap resampling
    - More computationally efficient for large datasets

3. ``'split'``: Split conformal
    - Simplest method
    - Uses a single train-calibration split
    - Less sample efficient

Select a method using the ``method`` parameter:

.. code-block:: python

    # Using CV+
    clf_cv = CoverForestClassifier(method='cv', cv=5)

    # Using Jackknife+-after-Bootstrap
    clf_boot = CoverForestClassifier(method='bootstrap')

    # Using split conformal
    clf_split = CoverForestClassifier(method='split', calib_size=0.3)

Advanced Features
-----------------

Parallel Processing
^^^^^^^^^^^^^^^^^^^

Enable parallel processing with the ``n_jobs`` parameter:

.. code-block:: python

    clf = CoverForestClassifier(n_jobs=-1)  # Use all cores
    reg = CoverForestRegressor(n_jobs=4)    # Use 4 cores

Sample Weights
^^^^^^^^^^^^^^

Both classes support sample weights:

.. code-block:: python

    clf.fit(X_train, y_train, sample_weight=weights)
    reg.fit(X_train, y_train, sample_weight=weights)

Custom Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^

Use custom cross-validation splitters with ``cv``:

.. code-block:: python

    from sklearn.model_selection import GroupKFold

    group_cv = GroupKFold(n_splits=5)
    clf = CoverForestClassifier(
        method='cv',
        cv=group_cv
    )

Next Steps
----------

- Check out the :doc:`API Reference <api>` for detailed documentation
- See :doc:`Examples <examples>` for more use cases
- Visit our `GitHub repository <https://github.com/mysite/coverforest>`_ for source code
