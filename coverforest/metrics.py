from numbers import Real

import numpy as np
from sklearn.utils._array_api import _average
from sklearn.utils._param_validation import (
    Interval,
    validate_params,
)
from sklearn.utils.validation import check_consistent_length, column_or_1d


@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],
        "y_pred": [tuple, "array-like", "sparse matrix"],
        "beta": [Interval(Real, 0.0, None, closed="both")],
        "labels": ["array-like", None],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def classification_coverage_score(y_true, y_pred, *, labels=None, sample_weight=None):
    """Compute the empirical coverage for classification prediction sets.

    The coverage score measures the proportion of true labels that are included
    in the prediction sets.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : tuple, list or array-like of shape (n_samples, n_classes)
        Binary matrix indicating the predicted set for each sample, where 1
        indicates the class is included in the prediction set and 0 indicates
        it is not.

    labels : array-like of shape (n_classes,), default=None
        List of labels in the same order of the columns of y_pred.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    score : float
        Returns the empirical coverage, i.e., the proportion of true labels
        included in the prediction sets, weighted by sample_weight.
        Best value is 1 and worst value is 0.

    Examples
    --------
    >>> import numpy as np
    >>> from metrics import classification_coverage_score
    >>> y_true = [0, 1, 2]
    >>> y_pred = np.array([[1, 0, 1], [0, 0, 1], [0, 0, 1]])
    >>> labels = [0, 1, 2]
    >>> classification_coverage_score(y_true, y_pred, labels=labels)
    0.66...
    """

    if isinstance(y_pred, tuple):
        y_pred = y_pred[1]

    y_true = column_or_1d(y_true)
    n = len(y_pred)
    assert y_true.shape[0] == n
    check_consistent_length(y_true, sample_weight)

    if isinstance(y_pred, list):
        is_in_y_pred = [y_true[i] in y_pred[i] for i in range(n)]
    else:
        if labels is None:
            raise ValueError("`labels` must be specified when `y_pred` is an array.")
        class_to_idx = {c: i for i, c in enumerate(labels)}
        y_idx = np.vectorize(class_to_idx.__getitem__)(y_true)
        is_in_y_pred = y_pred[np.arange(len(y_pred)), y_idx]

    return float(_average(is_in_y_pred, weights=sample_weight))


@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],
        "y_pred": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def average_set_size_loss(y_true, y_pred):
    """Compute the average size of classification prediction sets.

    For each sample, the set size is the number of classes included in
    the prediction set (sum of binary indicators).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : tuple, list or array-like of shape (n_samples, n_classes)
        Binary matrix indicating the predicted set for each sample, where 1
        indicates the class is included in the prediction set and 0 indicates
        it is not.

    Returns
    -------
    score : float
        Returns the average prediction set size.
        Minimum possible value is 0, maximum is n_classes.

    Examples
    --------
    >>> import numpy as np
    >>> from metrics import average_set_size_loss
    >>> y_pred = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    >>> average_set_size_loss(y_pred)
    1.333...
    """

    if isinstance(y_pred, tuple):
        y_pred = y_pred[1]

    if isinstance(y_pred, list):
        y_sizes = [len(y_pred) for y_pred in y_pred]
    else:
        y_sizes = y_pred.sum(axis=1)

    return float(_average(y_sizes))


@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],
        "y_pred": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def regression_coverage_score(y_true, y_pred, *, sample_weight=None):
    """Compute the empirical coverage for regression prediction intervals.

    The coverage score measures the proportion of true values that fall
    within the predicted intervals.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : tuple, list or array-like of shape (n_samples, 2)
        Predicted intervals, where each row contains [lower_bound, upper_bound].

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    score : float
        Returns the empirical coverage, i.e., the proportion of true values
        falling within the prediction intervals, weighted by sample_weight.
        Best value is 1 and worst value is 0.

    Examples
    --------
    >>> import numpy as np
    >>> from metrics import regression_coverage_score
    >>> y_true = [1.0, -2.0, 3.0]
    >>> y_pred = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    >>> regression_coverage_score(y_true, y_pred)
    0.66...
    """

    if isinstance(y_pred, tuple):
        y_pred = y_pred[1]

    y_true = column_or_1d(y_true)
    check_consistent_length(y_true, y_pred, sample_weight)

    low = y_pred[:, 0]
    high = y_pred[:, 1]
    return float(_average((low <= y_true) & (y_true <= high), weights=sample_weight))


@validate_params(
    {
        "y_true": ["array-like", "sparse matrix"],
        "y_pred": ["array-like", "sparse matrix"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def average_interval_length_loss(y_true, y_pred):
    """Compute the average length of regression prediction intervals.

    For each sample, the interval length is the difference between
    the upper and lower bounds.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : tuple, list or array-like of shape (n_samples, 2)
        Predicted intervals, where each row contains [lower_bound, upper_bound].

    Returns
    -------
    score : float
        Returns the average interval length.
        Minimum possible value is 0, no maximum value.

    Examples
    --------
    >>> import numpy as np
    >>> from metrics import average_interval_length_loss
    >>> y_pred = np.array([[0.5, 2.5], [1.5, 4.5], [2.5, 3.5]])
    >>> average_interval_length_loss(y_pred)
    2.0
    """

    if isinstance(y_pred, tuple):
        y_pred = y_pred[1]

    low = y_pred[:, 0]
    high = y_pred[:, 1]
    return float(_average(high - low))
