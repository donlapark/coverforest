"""
Conformal Random Forest methods.

Those methods include four different implementations of random forests that use
conformal prediction methods to output set predictions.

The module structure is the following:

The ``CoverForestClassifier`` and ``CoverForestRegressor`` classes each provide three
methods random forest classifiers that output prediction sets and prediction
intervals using conformal prediction.

    - For classification, the prediction sets are obtained using adaptive
      prediction set (APS) proposed by Romano, Sesia & Candès (2020). The code also
      provides regularization introduced by Angelopoulos, Bates, Malik & Jordan
      (2021) to encourage smaller sets.
    - For regression, the prediction intervals are obtained using the Jackknife+ and
      CV+ on the residuals proposed by Barber, Candès, Ramdas & Tibshirani (2021).

The method, specified in the ``method`` parameter, includes:

    - ``method=cv``:   Random Forest with CV+ for prediction sets/intervals
    - ``method=bootstrap``:   Random Forest with Jackknife+-after-Bootstrap for
      prediction sets/intervals
    - ``method=split``:   Random Forest with split adaptive prediction set
      (APS) for prediction sets/intervals
"""

# scikit-learn authors: Gilles Louppe <g.louppe@gmail.com>
#                       Brian Holt <bdholt1@gmail.com>
#                       Joly Arnaud <arnaud.v.joly@gmail.com>
#                       Fares Hedayati <fares.hedayati@gmail.com>
#  coverforest authors: Donlapark Ponnoprat <donlapark.p@cmu.ac.th>
#                       Panisara Meehinkong
#
# License: BSD 3 clause

from abc import abstractmethod
from copy import deepcopy
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn

import numpy as np
from scipy.sparse import issparse
from sklearn.base import _fit_context, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (
    BaseForest,
    ForestClassifier,
    ForestRegressor,
    _get_n_samples_bootstrap,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils import compute_sample_weight
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._tags import ClassifierTags, RegressorTags
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    _check_y,
    check_consistent_length,
    check_is_fitted,
    check_random_state,
    validate_data,
)

from ._fast_random_forest import (
    BaseFastForest,
    FastRandomForestClassifier,
    FastRandomForestRegressor,
)
from ._giqs import _compute_predictions_split, _compute_test_giqs_cv
from .metrics import (
    average_interval_length_loss,
    average_set_size_loss,
    classification_coverage_score,
    regression_coverage_score,
)

MAX_INT = np.iinfo(np.int32).max


def _generate_sample_indices(
    random_state, kfold_indices, k, tree_idx, n_samples, n_samples_bootstrap, method
):
    """Generate sample indices for building trees.

    Private function used to generate sample indices for individual trees in parallel
    forest construction.

    Parameters
    ----------
    random_state : int, RandomState instance or None.
        The random number generator instance.
    kfold_indices : list of tuples of ndarrays or None
        Pairs of train-test indices obtained from k-fold cross-validation.
    k : int or None
        Current fold number.
    tree_idx : int or None
        Index of the current tree.
    n_samples : int
        Total number of samples.
    n_samples_bootstrap : int
        Number of samples to draw for bootstrap.
    method : {'cv', 'bootstrap', 'split'}
        The method used for generating sample indices.

    Returns
    -------
    sample_indices : ndarray of shape (n_samples_bootstrap,)
        The generated sample indices for building the tree.
    """

    if method == "cv":
        sample_indices = kfold_indices[tree_idx % k][0]
    else:
        random_instance = check_random_state(random_state)
        sample_indices = random_instance.randint(
            0, n_samples, n_samples_bootstrap, dtype=np.int32
        )

    return sample_indices


def _parallel_build_trees(
    tree,
    X,
    y,
    method,
    kfold_indices,
    k,
    sample_weight,
    tree_idx,
    n_trees,
    oob_mat,
    verbose=0,
    class_weight=None,
    n_samples=None,
    n_samples_bootstrap=None,
    random_state=None,
    n_classes=None,
):
    """Fit a single tree in parallel on a subsample obtained via `method`.

    Parameters
    ----------
    tree : BaseDecisionTree
        The decision tree instance to be built.
    X : ndarray of shape (n_samples, n_features)
        The training input samples.
    y : ndarray of shape (n_samples,)
        The target values.
    method : {'cv', 'bootstrap', 'split'}
        Subsampling method used for conformal predictions.
    kfold_indices : list of tuples of ndarrays
        Used when `method='cv'`. Pairs of train-test indices obtained from
        k-fold cross-validation.
    k : int
        Used when `method='cv'`. Current fold number.
    sample_weight : array-like of shape (n_samples,) or None
        Sample weights.
    tree_idx : int
        Index of the current tree.
    n_trees : int
        Total number of trees to be built.
    verbose : int, default=0
        Controls the verbosity of the tree building process.
    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Class weights.
    n_samples : int or None, default=None
        Number of samples in the dataset.
    n_samples_bootstrap : int or None, default=None
        Used when `method='bootstrap'`. Number of samples to draw for bootstrap.
    n_classes : int or None, default=None
        Number of classes in the dataset.

    Returns
    -------
    tree : BaseDecisionTree
        The fitted decision tree.
    """

    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if sample_weight is None:
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()

    indices = _generate_sample_indices(
        tree.random_state,
        kfold_indices,
        k,
        tree_idx,
        n_samples,
        n_samples_bootstrap,
        method,
    )

    sample_counts = np.bincount(indices, minlength=n_samples)

    oob_mat[:, tree_idx] = sample_counts == 0

    curr_sample_weight *= sample_counts

    if not hasattr(tree, "n_classes_"):
        tree.n_classes_ = n_classes

    if class_weight == "subsample":
        with catch_warnings():
            simplefilter("ignore", DeprecationWarning)
            curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
    elif class_weight == "balanced_subsample":
        curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

    tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)

    return tree


def _accumulate_prediction(predict, X, i, out):
    """Store the test predictions of each tree in an array.

    This method will be called in parallel.
    """

    out[i] = predict(X, check_input=False)


class ConformalClassifierMixin:
    """Mixin class for conformal classifiers in scikit-learn.

    This mixin defines the following functionality:

    - set estimator type to `"classifier"` through the `estimator_type` tag;
    - `score` method that evaluates both coverage and average set size;
    - enforce that `fit` requires `y` to be passed through the `requires_y` tag.

    Notes
    -----
    Conformal classifiers output prediction sets C(X) that satisfy:

        P(Y ∈ C(X)) ≥ 1 - α

    where:

    - α is the desired miscoverage rate
    - The probability is taken over the pair (X,Y)
    - This guarantee holds for any data distribution, with an assumption that
      the samples (X₁,Y₁), ... , (Xₙ,Yₙ) are i.i.d. Some methods require a
      weaker assumption that they are exchangable.

    The score method supports different metrics:

    - 'coverage': Empirical coverage probability
    - 'size': Average size of prediction sets
    - 'both': Returns both coverage and average size.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from coverforest import ConformalClassifierMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(ConformalClassifierMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y):
    ...         self.is_fitted_ = True
    ...         self.n_classes_ = np.unique(y).shape[0]
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=(X.shape[0], self.n_classes_),
                             fill_value=self.param)
    >>> estimator = MyEstimator(param=1)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([[1, 1], [1, 1], [1, 1]])
    >>> clf.score(X, y, alpha=0.05, scoring='both')
    (0.66, 2)...


    References
    ----------
    .. [1] Yaniv Romano, Matteo Sesia, Emmanuel J. Candès. Classification with
           Valid and Adaptive Coverage. NeurIPS 2020.
    """

    _estimator_type = "classifier"

    def score(self, X, y, alpha=0.05, scoring="coverage", sample_weight=None):
        """Evaluate the prediction set on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        alpha : float, default=None
            The desired miscoverage rate. The method will construct prediction sets
            with approximately (1-alpha) coverage.

        scoring : {'size', 'coverage', 'both'}, default='size'
            The scoring metric to use:

            - 'size': returns the average size of prediction sets
            - 'coverage': returns the empirical coverage (proportion of sets that
              contain true labels)
            - 'both': returns a tuple of (coverage, average_size)

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        score : float or tuple
            If scoring='size': returns average prediction set size (float)
            If scoring='coverage': returns empirical coverage (float)
            If scoring='both': returns (coverage, average_size) tuple.
        """

        y = _check_y(y, estimator=self)
        check_consistent_length(X, y, sample_weight)
        _, y_set = self.predict(X, alpha, binary_output=True)

        if scoring == "size" or scoring == "both":
            avg_size = average_set_size_loss(y, y_set)
            if scoring == "size":
                return avg_size
        if scoring == "coverage" or scoring == "both":
            coverage = classification_coverage_score(
                y, y_set, labels=self.classes_, sample_weight=sample_weight
            )
            if scoring == "coverage":
                return coverage
        if scoring == "both":
            return (coverage, avg_size)
        else:
            raise ValueError("`scoring` must be one of `coverage`, `size` or `both`.")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.non_deterministic = True
        tags.input_tags.allow_nan = True
        tags.target_tags.required = True
        tags.target_tags.single_output = True
        tags.target_tags.multi_output = False
        return tags


class ConformalRegressorMixin:
    """Mixin class for conformal regressors in scikit-learn.

    This mixin defines the following functionality:

    - set estimator type to `"regressor"` through the `estimator_type` tag;
    - `score` method that evaluates both coverage and average interval length;
    - enforce that `fit` requires `y` to be passed through the `requires_y` tag.

    Notes
    -----
    Conformal regressors output prediction intervals [l(X), u(X)] that satisfy:

        P(l(X) ≤ Y ≤ u(X)) ≥ 1 - α

    where:

    - α is the desired miscoverage rate
    - The probability is taken over the pair (X,Y)
    - This guarantee holds for any data distribution with an assumption that
      the samples (X₁,Y₁), ... , (Xₙ,Yₙ) are i.i.d. Some methods require a
      weaker assumption that they are exchangable.

    The score method supports different metrics:

    - 'coverage': Empirical coverage probability
    - 'length': Average length of prediction intervals
    - 'both': Returns both coverage and length metrics

    The intervals can be constructed using the three methods:
    - Jackknife+
    - CV+
    - Jackknife+-after-Bootstrap

    Each method provides valid coverage guarantees.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from coverforest import ConformalClassifierMixin
    >>> class MyConformalRegressor(ConformalRegressorMixin, BaseEstimator):
    ...     def predict(self, X, alpha=0.05):
    ...         n_samples = X.shape[0]
    ...         y_pred = np.zeros(n_samples)
    ...         intervals = np.column_stack([-np.ones(n_samples),
    ...                                     np.ones(n_samples)])
    ...         return y_pred, intervals
    >>> reg = MyConformalRegressor()
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([0.5, 1.5])
    >>> reg.score(X, y, alpha=0.05, scoring='both')
    (0.5, 2)...

    References
    ----------
    .. [1] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and
           Ryan J. Tibshirani. "Predictive inference with the jackknife+."
           Ann. Statist., 49(1):486–507, (2021).
    .. [2] Byol Kim, Chen Xu, Rina Foygel Barber. Predictive inference is free
           with the jackknife+-after-bootstrap. NeurIPS 2020.
    """

    _estimator_type = "regressor"

    def score(self, X, y, alpha=0.05, scoring="coverage", sample_weight=None):
        """Evaluate the prediction intervals on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        alpha : float, default=None
            The desired miscoverage rate. The method will construct prediction
            intervals with approximately (1-alpha) coverage.

        scoring : {'length', 'coverage', 'both'}, default='length'
            The scoring metric to use:

            - 'length': returns the average length of prediction intervals
            - 'coverage': returns the empirical coverage (proportion of true values in
            intervals)
            - 'both': returns a tuple of (coverage, average_length)

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        score : float or tuple
            If scoring='length': returns average interval length (float)
            If scoring='coverage': returns empirical coverage (float)
            If scoring='both': returns (coverage, average_length) tuple
        """

        y = _check_y(y, estimator=self)
        check_consistent_length(X, y, sample_weight)
        _, y_intervals = self.predict(X, alpha)

        if scoring == "length" or scoring == "both":
            avg_length = average_interval_length_loss(y, y_intervals)
            if scoring == "length":
                return avg_length
        if scoring == "coverage" or scoring == "both":
            coverage = regression_coverage_score(
                y, y_intervals, sample_weight=sample_weight
            )
            if scoring == "coverage":
                return coverage
        if scoring == "both":
            return (coverage, avg_length)
        else:
            raise ValueError(
                "`scoring` must be one of `'coverage'`, `'length'` or `'both'`."
            )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.non_deterministic = True
        tags.input_tags.allow_nan = True
        tags.target_tags.required = True
        tags.target_tags.single_output = True
        tags.target_tags.multi_output = False
        return tags


class BaseConformalForest(BaseForest):
    """Base class for conformal forests of trees.

    This class extends scikit-learn's BaseForest to implement conformal
    prediction.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    _parameter_constraints = {
        **BaseForest._parameter_constraints,
        "method": [StrOptions({"bootstrap", "cv", "split"})],
        "cv": [Interval(Integral, 2, None, closed="left"), "cv_object"],
        "n_forests_per_fold": [Interval(Integral, 1, None, closed="left")],
        "resample_n_estimators": ["boolean"],
    }

    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        estimator_params=tuple(),
        n_estimators=100,
        method="cv",
        cv=5,
        n_forests_per_fold=1,
        resample_n_estimators=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.n_estimators = n_estimators
        self.method = method
        self.cv = cv
        self.n_forests_per_fold = n_forests_per_fold
        self.resample_n_estimators = resample_n_estimators
        self.class_weight = class_weight

    def _check_data(self, X, y=None, sample_weight=None, expanded_class_weight=None):
        """Validate or convert input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        expanded_class_weight : array-like of shape (n_samples,), default=None
            Expanded class weights computed from `class_weight` parameter.

        Returns
        -------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The validated and converted feature matrix.

        y : array-like of shape (n_samples,)
            The validated target values.

        sample_weight : array-like of shape (n_samples,)
            The validated sample weights.
        """

        if y is None:
            raise ValueError(
                "The model requires y to be passed, but the target y is None."
            )
        else:
            if issparse(y):
                raise ValueError("sparse multilabel-indicator for y is not supported.")
            X, y = validate_data(
                self,
                X,
                y,
                accept_sparse="csc",
                dtype=DTYPE,
                ensure_min_samples=2,
                ensure_all_finite=False,
            )

        sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        if y is not None:
            y = np.atleast_1d(y)
            if y.ndim == 2 and y.shape[1] == 1:
                y = np.ravel(y)

            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is "
                        "not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not strictly positive which "
                        "is necessary for Poisson regression."
                    )

            self._n_samples, self.n_outputs_ = y.shape

        return X, y, sample_weight

    def _fit_wrapper(self, X, y, calib_size, sample_weight):
        """Fit the forest using either CV+ or split conformal method.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        calib_size : float
            The proportion of samples to use for calibration in split conformal.
            Only used when method='split'.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        if self.method in ["cv", "bootstrap"]:
            self._fit_cv(X, y, self.cv, sample_weight)
        else:
            self._fit_split(X, y, calib_size, sample_weight)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def _fit_cv(self, X, y, cv, sample_weight):
        """ "Build a forest of trees using CV+ or Jackknife+-after-bootstrap.

        This method will be called when calling the `fit` method with
        `method='cv'` or `method='bootstrap'`.

        - If `method='cv'`, the data will be split into K folds and the trees
          will be fitted on K-1 folds.
        - If `method='bootstrap'`, the trees will be fitted on bootstrap
        subsamples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        cv : int or cross-validation generator
            Determines the cross-validation splitting strategy. If int,
            specifies the number of folds. See scikit-learn's model selection
            module for available cross-validation objects. Only used when
            `method='cv'`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y, sample_weight = self._check_data(X, y, sample_weight)
        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        self.y_ = y
        random_state = check_random_state(self.random_state)

        if self.method == "bootstrap":
            self._n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=self._n_samples, max_samples=self.max_samples
            )
            binom_prop = np.exp(
                self._n_samples_bootstrap * np.log(1 - 1 / (self._n_samples + 1))
            )
            if self.resample_n_estimators:
                n_estimators = np.maximum(
                    2,
                    random_state.binomial(self.n_estimators / binom_prop, binom_prop),
                )
            else:
                n_estimators = self._n_samples_bootstrap

            self._n_cv_folds = None
        else:
            if isinstance(cv, Integral):
                if cv <= 1:
                    raise ValueError(
                        "k-fold cross-validation requires at least one"
                        " train/test split by setting n_splits=2 or more,"
                        " got n_splits={0}.".format(cv)
                    )

                if cv > self._n_samples:
                    print(
                        f"The number of splits of {self._n_samples} is too low. Set the"
                        " number of splits as the number of samples instead."
                    )
                    cv = self._n_samples

                self._n_cv_folds = cv

                cv = KFold(
                    n_splits=self._n_cv_folds, shuffle=True, random_state=random_state
                ).split(X)
                cv = list(cv)
            else:
                self._n_cv_folds = len(cv)

            n_estimators = self.n_forests_per_fold * self._n_cv_folds

            self._n_samples_bootstrap = None

        self.kfold_indices_ = cv
        self._validate_estimator()

        if self.method != "bootstrap" and self.oob_score:
            raise ValueError("Out of bag estimation only available if method=oob")

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        self.oob_matrix_ = np.zeros((self._n_samples, n_estimators), dtype=bool)
        n_more_estimators = n_estimators - len(self.estimators_)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        if is_classifier(self) and hasattr(self, "n_classes_"):
            oob_pred_shape = (self._n_samples, self.n_classes_, self.n_outputs_)
            n_classes = self.n_classes_
        else:
            oob_pred_shape = (self._n_samples, 1, self.n_outputs_)
            n_classes = 1

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            if issparse(X):
                X_csr = X.tocsr()
            else:
                X_csr = None

            trees = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads"
            )(
                delayed(_parallel_build_trees)(
                    t,
                    X,
                    y,
                    self.method,
                    self.kfold_indices_,
                    self._n_cv_folds,
                    sample_weight,
                    i,
                    len(trees),
                    self.oob_matrix_,
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples=self._n_samples,
                    n_samples_bootstrap=self._n_samples_bootstrap,
                    n_classes=n_classes,
                )
                for i, t in enumerate(trees)
            )
            # Collect newly grown trees
            self.estimators_.extend(trees)
        if self.oob_score and (
            n_more_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, binary. "
                )

        # calculate oob scores
        oob_pred = np.zeros(oob_pred_shape, dtype=np.float64)

        pred_func = "predict_proba" if is_classifier(self) else "predict"

        for i, tree in enumerate(self.estimators_):
            predict_func = getattr(tree, pred_func)

            if X_csr is not None:
                y_pred = predict_func(
                    X_csr[self.oob_matrix_[:, i], :], check_input=False
                )
            else:
                y_pred = predict_func(X[self.oob_matrix_[:, i], :], check_input=False)

            y_pred = y_pred.reshape(-1, n_classes, 1)
            oob_pred[self.oob_matrix_[:, i], ...] += y_pred

        self._n_oob_pred = self.oob_matrix_.sum(axis=1, keepdims=True)
        if (self._n_oob_pred == 0).any():
            warn(
                (
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates."
                ),
                UserWarning,
            )
            self._n_oob_pred[self._n_oob_pred == 0] = 1
        oob_pred /= self._n_oob_pred[:, None, :]
        self.oob_pred_ = oob_pred

        if is_classifier(self) and self.k_star_ is not None:
            self.train_giqs_ = self._compute_train_giqs(oob_pred, self.y_)

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def _fit_split(self, X, y, calib_size, sample_weight):
        """Build a forest of trees using split conformal prediction.

        This method will be called when calling the `fit` method with
        `method='split'`.

        The training set will be split into a smaller training set and a
        calibration set. The model will be fitted on the former and calibration
        scores will be calculated on the latter.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        calib_size : float
            The proportion of samples to use for calibration. Should be in
            (0, 1).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y, sample_weight = self._check_data(X, y, sample_weight)

        if hasattr(self, "feature_names_in_"):
            feature_names_in_ = self.feature_names_in_
        else:
            feature_names_in_ = None

        random_state = check_random_state(self.random_state)

        if is_classifier(self):
            y, expanded_class_weight = self._validate_y_class_weight(y)

            if expanded_class_weight is not None:
                if sample_weight is not None:
                    sample_weight = sample_weight * expanded_class_weight
                else:
                    sample_weight = expanded_class_weight

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        sample_weight_int = sample_weight.astype(int)
        is_integer_weights = np.all(np.isclose(sample_weight, sample_weight_int, 1e-10))

        if is_integer_weights:
            sample_weight = sample_weight_int
            rep_indices = np.repeat(np.arange(X.shape[0]), sample_weight)
            X = X[rep_indices]
            y = y[rep_indices]
            sample_weight = np.ones(len(y))

        stratify = y if is_classifier(self) else None

        X_train, X_cal, y_train, y_cal, sample_weight_train, sample_weight_cal = (
            train_test_split(
                X,
                y,
                sample_weight,
                test_size=calib_size,
                stratify=stratify,
                random_state=random_state,
            )
        )

        BaseFastForest.fit(self, X_train, y_train, sample_weight=sample_weight_train)

        if self.oob_score and not hasattr(self, "oob_score_"):
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )

        self.feature_names_in_ = None
        if is_classifier(self):
            self.oob_pred_ = self.predict_proba(X_cal)[:, :, None]
            if hasattr(self, "classes_") and self.n_outputs_ == 1:
                self.n_classes_ = self.n_classes_[0]
                self.classes_ = self.classes_[0]
        else:
            self.oob_pred_ = FastRandomForestRegressor.predict(self, X_cal)

        self.feature_names_in_ = feature_names_in_

        if y_cal.ndim == 1:
            y_cal = np.reshape(y_cal, (-1, 1))
        self.y_ = y_cal
        self._n_samples = self.y_.shape[0]

        if is_classifier(self) and self.k_star_ is not None:
            self.train_giqs_ = self._compute_train_giqs(self.oob_pred_, self.y_)

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def _compute_oob_predictions(self, X):
        """Compute out-of-bag predictions on the test set `X`.

        This method will be called during the prediction with `method='cv'` or
        `method='bootstrap'`. It provides each training point the average
        predictions (or probability predictions for classification) of its
        out-of-bag trees on `X`.

        The output of this method will be used to calculate the conformity
        scores of each point in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray
            The predicted values.

        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs) or /
                (n_samples, 1, n_outputs)
            The out-of-bag predictions.
        """

        # Prediction requires X to be in CSR format
        if issparse(X):
            X = X.tocsr()

        n_samples = X.shape[0]
        n_estimators = len(self.estimators_)
        classifier = is_classifier(self)

        if classifier and hasattr(self, "n_classes_"):
            n_classes = self.n_classes_
        else:
            n_classes = 1

        if classifier:
            y_pred_shape = (n_samples, n_classes)
            y_pred_all_shape = (n_estimators, n_samples, n_classes)
        else:
            y_pred_shape = (n_samples,)
            y_pred_all_shape = (n_estimators, n_samples)

        y_pred = np.zeros(y_pred_shape, dtype=np.float64)
        y_pred_all = np.zeros(y_pred_all_shape, dtype=np.float64)

        n_jobs, _, _ = _partition_estimators(n_estimators, self.n_jobs)
        classifier = is_classifier(self)
        pred_func = "predict_proba" if classifier else "predict"

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(getattr(e, pred_func), X, i, y_pred_all)
            for i, e in enumerate(self.estimators_)
        )

        if classifier:
            oob_pred = self.oob_matrix_ @ y_pred_all.reshape(n_estimators, -1)
            oob_pred = oob_pred.reshape(self._n_samples, n_samples, n_classes)
            oob_pred /= self._n_oob_pred[:, :, None]
        else:
            oob_pred = self.oob_matrix_ @ y_pred_all / self._n_oob_pred

        y_pred = y_pred_all.mean(axis=0)

        return y_pred, oob_pred

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the base estimator.

        This is a modification of sklearn's `_make_estimator` method only when
        `method='cv'` to fix the problem of model's parameters not being
        passed to the `FastRandomForest` sub-estimators correctly.

        Parameters
        ----------
        append : bool, default=True
            If True, append the estimator to the list of estimators.

        random_state : int, RandomState instance or None, default=None
            Controls the random seed used to initialize the base estimator.

        Returns
        -------
        estimator : object
            The configured base estimator.
        """

        if self.method == "cv":
            estimator = deepcopy(self.estimator)
            estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

            if random_state is not None:
                random_state = check_random_state(random_state)
                to_set = {}
                for key in sorted(estimator.get_params(deep=True)):
                    if key == "random_state" or key.endswith("__random_state"):
                        to_set[key] = random_state.randint(np.iinfo(np.int32).max)

                if to_set:
                    estimator.set_params(**to_set)
            if append:
                self.estimators_.append(estimator)

        else:
            estimator = super()._make_estimator(append, random_state)

        return estimator

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The validated input.

        Raises
        ------
        ValueError
            If sparse matrix has wrong dtype for indices/indptr.
        """

        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            dtype=DTYPE,
            accept_sparse="csr",
            reset=False,
            ensure_all_finite="allow-nan",
        )
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based sparse matrices")
        return X


class ConformalForestClassifier(
    ConformalClassifierMixin, BaseConformalForest, ForestClassifier
):
    """
    Base class for forest of conformal trees-based classifiers.

    This class extends scikit-learn's ForestClassifier to implement conformal
    prediction.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        n_estimators=100,
        method="cv",
        cv=5,
        k_init="auto",
        lambda_init="auto",
        repeat_params_search=True,
        allow_empty_sets=True,
        randomized=True,
        alpha_default=None,
        n_forests_per_fold=1,
        resample_n_estimators=True,
        estimator_params=tuple(),
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            estimator=estimator,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            method=method,
            cv=cv,
            n_forests_per_fold=n_forests_per_fold,
            resample_n_estimators=resample_n_estimators,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.k_init = k_init
        self.lambda_init = lambda_init
        self.repeat_params_search = repeat_params_search
        self.randomized = randomized
        self.allow_empty_sets = allow_empty_sets
        self.alpha_default = alpha_default

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, alpha=0.05, calib_size=0.3, valid_size=0.3, sample_weight=None):
        """Fit the conformal forest classifier.

        This method first finds optimal values for the regularization
        parameters k and lambda using the procedure from Angelopoulos et al.
        (2021), then fits the forest using either CV+,
        Jackknife+-after-Bootstrap, or split conformal prediction.

        The k parameter penalizes prediction sets containing more than k
        classes, while lambda controls the strength of this penalty. When
        `k_init='auto'` and/or `lambda_init='auto'`, these parameters are chosen
        automatically:

        - k is set to the (1-alpha)-quantile of the rank of true labels in the
          out-of-bag predictions
        - lambda is selected from `[0.001, 0.01, 0.1, 0.2, 0.5, 1]` using a
          held-out validation set to minimize prediction set size

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values (class labels).

        alpha : float, default=0.05
            Desired miscoverage rate used to search for the optimal `k` and
            `lambda`. The prediction sets will be constructed to contains the
            true label with probability at least 1-alpha.

            This parameter will not be used when both `k_init` and
            `lambda_init` are both numeric values.

        calib_size : float, default=0.3
            Used when method='split'. The proportion of training samples to use
            for calibration.

        valid_size : float, default=0.3
            The proportion of samples to use for validation when searching for
            the optimal lambda parameter.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.

        References
        ----------
        .. [1] Anastasios Nikolas Angelopoulos, Stephen Bates,
               Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
               Classifiers using Conformal Prediction", ICLR 2021.
        """

        self.search_k_and_lambda(X, y, alpha, calib_size, valid_size, sample_weight)

        if self.verbose:
            print(f"Fitting with k = {self.k_star_} and lambda = {self.lambda_star_}.")

        self._fit_wrapper(X, y, calib_size, sample_weight)

        return self

    def search_k_and_lambda(
        self, X, y, alpha=0.05, calib_size=0.3, valid_size=0.3, sample_weight=None
    ):
        """Search for optimal values of k and lambda parameters and store them as
        attributes `k_star_` and `lambda_star_`, respectively.

        The parameter search follows the procedure suggested by Angelopoulos,
        Bates, Jordan & Jitendra Malik (2021):

        - `k_star_` is the (1-alpha)-quantile of the rank of true y in the
          out-of-bag predictions
        - `lambda_star_` is chosen from the candidates
          `[0.001, 0.01, 0.1, 0.2, 0.5, 1]` using a held-out validation set.

        The parameter search be performed only when constructing the model with
        `lambda_init='auto'` and/or `k_init='auto'`, otherwise they will be set
        to the provided values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        alpha : float, default=0.05
            The desired miscoverage rate.

        calib_size : float, default=0.3
            Used when `method='split'`. The proportion of training samples to
            use for split conformal prediction.

        valid_size : float, default=0.3
            The proportion of samples to use for validation of lambda.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        k_star : int
            The value of k obtained from the (1-alpha) of the observed ranks of
            the training targets.

        lambda_star : float
            The value of lambda obtained from the grid search cross-validation.

        References
        ----------
        .. [1] Anastasios Nikolas Angelopoulos, Stephen Bates,
               Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
               Classifiers using Conformal Prediction", ICLR 2021.
        """

        if self.k_init == "auto":
            if self.repeat_params_search or not hasattr(self, "k_star_"):
                self.k_star_ = None
        else:
            self.k_star_ = self.k_init

        if self.lambda_init == "auto":
            if self.repeat_params_search or not hasattr(self, "lambda_star_"):
                self.lambda_star_ = None
        else:
            self.lambda_star_ = self.lambda_init

        if self.k_star_ is None or self.lambda_star_ is None:
            if self.verbose:
                print("Searching regularization parameters...")
            X, y, sample_weight = self._check_data(X, y, sample_weight)
            if y.ndim == 2 and y.shape[1] == 1:
                y = np.ravel(y)

            random_state = check_random_state(self.random_state)
            random_state = check_random_state(random_state.randint(MAX_INT))

            if self.k_star_ is None:
                self._fit_wrapper(X, y, calib_size, sample_weight)
                true_label_scores = self.oob_pred_[
                    np.arange(len(self.y_)), self.y_.flatten().astype(int)
                ]
                scores_compare = true_label_scores < self.oob_pred_[:, :, 0]
                y_ranks = scores_compare.sum(axis=1).ravel()
                k_star = np.quantile(y_ranks, 1 - alpha, method="higher") + 1
                self.k_star_ = k_star

            if self.lambda_star_ is None:
                X1, X2, y1, _, sw1, _ = train_test_split(
                    X,
                    y,
                    sample_weight,
                    test_size=valid_size,
                    random_state=random_state,
                    stratify=y,
                )
                best_sum_size = MAX_INT
                for lambda_ in [0.001, 0.01, 0.1, 0.2, 0.5, 1]:
                    self.lambda_star_ = lambda_
                    self._fit_wrapper(X1, y1, calib_size, sw1)
                    _, y_set_pred = self.predict(X2, alpha=alpha, binary_output=True)
                    sum_size = y_set_pred.sum()
                    if sum_size < best_sum_size:
                        best_sum_size = sum_size
                        lambda_star = lambda_

                self.lambda_star_ = lambda_star

        k_star = self.k_star_
        lambda_star = self.lambda_star_

        return k_star, lambda_star

    def _compute_train_giqs(self, oob_pred, y):
        """Compute the regularized generalized inverse quantile conformity
        scores (giqs) of the training samples.

        Given a training sample with its out-of-bag probability predictions
        sorted in descending order π̂₁ ≥ ... ≥ π̂ₗ, if the sample's true label is
        j, then its (non-regularized) giqs is obtained by first sampling U from
        Uniform[0, 1] and then calculate::

            π̂₁ + ... + π̂ⱼ₋₁ + U * π̂ⱼ.

        The regularized giqs is then obtained by adding a term that penalizes
        sets by how much their sizes are larger than k::

            π̂₁ + ... + π̂ⱼ₋₁ + U * π̂ⱼ + λ * max(0, j - k + 1).

        Parameters
        ----------
        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs)
            The out-of-bag predictions of the training samples.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The true target values of the training samples.

        Returns
        -------
        train_giqs : ndarray of shape (n_samples, n_classes)
            The generalized inverse quantile scores of the training samples.

        Notes
        -----
        This method only supports `n_outputs == 1` at the moment.

        References
        ----------
        .. [1] Yaniv Romano, Matteo Sesia & Emmanuel J. Candès, "Classification
               with Valid and Adaptive Coverage", NeurIPS 2020.
        .. [2] Anastasios Nikolas Angelopoulos, Stephen Bates,
               Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
               Classifiers using Conformal Prediction", ICLR 2021.
        """

        y_score = oob_pred[np.arange(len(y)), y.flatten().astype(int)]
        pred_probs = oob_pred[:, :, 0]
        y_compare = y_score < pred_probs
        y_rank = y_compare.sum(axis=1).ravel()
        tau_before_y = (y_compare * pred_probs).sum(axis=1)
        tau = tau_before_y + y_score[:, 0]

        penalty = np.zeros((y.shape[0], y.shape[1]))
        penalty[:, self.k_star_ :] += self.lambda_star_

        if not self.randomized:
            train_giqs = tau + penalty[:, 0]
        else:
            train_giqs = np.zeros(tau.shape)
            rng = check_random_state(self.random_state)
            U = rng.random(size=tau.shape[0])

            # Decide whether to keep U; y might be removed if y_rank == 0
            zero_idx = y_rank == 0
            if not self.allow_empty_sets:
                train_giqs[zero_idx] = tau[zero_idx] + penalty[zero_idx, 0]
            else:
                train_giqs[zero_idx] = (
                    U[zero_idx] * tau[zero_idx] + penalty[zero_idx, 0]
                )
            non_zero_idx = np.arange(y.shape[0])[~zero_idx]
            sum_penalty = np.array(
                [penalty[i, 0 : (y_rank[i] + 1)].sum() for i in non_zero_idx]
            )
            train_giqs[non_zero_idx] = (
                U[non_zero_idx] * y_score[:, 0][non_zero_idx]
                + tau_before_y[non_zero_idx]
                + sum_penalty
            )

        return train_giqs

    def _compute_test_giqs(self, oob_pred, num_threads=4):
        """Compute the regularized generalized inverse quantile comformity
        scores (giqs) of test samples.

        This method will be called during prediction when `method='cv'`.

        Given a test sample with its out-of-bag probability predictions sorted
        in descending order π̂₁ ≥ ... ≥ π̂ₗ, the sample's (non-regularized) giqs
        for a label j is obtained by first sampling U from Uniform[0, 1] and
        then calculate::

            π̂₁ + ... + π̂ⱼ₋₁ + U * π̂ⱼ.

        The regularized giqs is then obtained by adding a term that penalizes
        prediction sets that are larger than k::

            π̂₁ + ... + π̂ⱼ₋₁ + U * π̂ⱼ + λ * max(0, j - k + 1).

        This method will be called during prediction when `method='cv'` or
        `method='bootstrap'`.

        Parameters
        ----------
        oob_pred : ndarray of shape (n_train, n_classes, n_test)
            The out-of-bag predictions of the test data.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        test_giqs : ndarray of shape (n_train, n_classes, n_test)
            The generalized inverse quantile conformity scores for test data.

        References
        ----------
        .. [1] Yaniv Romano, Matteo Sesia & Emmanuel J. Candès, "Classification
               with Valid and Adaptive Coverage", NeurIPS 2020.
        .. [2] Anastasios Nikolas Angelopoulos, Stephen Bates,
               Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
               Classifiers using Conformal Prediction", ICLR 2021.
        """

        test_giqs = np.zeros_like(oob_pred, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        seed = random_state.get_state()[1][0]
        _compute_test_giqs_cv(
            oob_pred,
            test_giqs,
            self.k_star_,
            self.lambda_star_,
            self.randomized,
            self.allow_empty_sets,
            num_threads,
            seed,
        )

        return test_giqs

    def _predict_from_giqs(self, y_pred, tau, num_threads=4):
        """Construct prediction sets from calibration set's generalized inverse
        quantile comformity scores (giqs). The construction follows Algorithm 3
        in Angelopoulos, Bates, Jordan & Malik (2021).

        For each sample, given probability predictions π̂₁ ≥ ... ≥ π̂ₗ sorted
        in descending order, the method first finds the label L whose
        regularized cumulative probabilities is just above τ::

            τ ≤ π̂₁ + ... + π̂ʟ + λ * max(0, L - k + 1) ≤ τ + 1,

        where k is the regularization parameter.

        Define π̂ʟ' = π̂ʟ + λ if L ≥ k, otherwise π̂'ʟ = π̂ʟ. Sample  U from
        Uniform[0, 1]. The prediction set is {1, ... , L - 1} if

            π̂₁ + ... + π̂ʟ + λ * max(0, L - k + 1) - τ ≤ U * π̂ʟ',

        otherwise the prediction is {1, ... , L}.

        This method will be called during prediction when `method='split'`.

        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_classes)
            The probability predictions of the test set.

        tau : float, default=None
            The generalized inverse quantile of the calibration set.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        y_set_pred : ndarray of shape (n_samples, n_classes)
            Prediction sets of the test data as a binary array where 1
            indicates class membership in the set.

        References
        ----------
        .. [1] Yaniv Romano, Matteo Sesia & Emmanuel J. Candès, "Classification
               with Valid and Adaptive Coverage", NeurIPS 2020.
        .. [2] Anastasios Nikolas Angelopoulos, Stephen Bates,
               Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
               Classifiers using Conformal Prediction", ICLR 2021.
        """

        y_set_pred = np.zeros_like(y_pred, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        seed = random_state.get_state()[1][0]
        _compute_predictions_split(
            y_pred,
            y_set_pred,
            tau,
            self.k_star_,
            self.lambda_star_,
            self.randomized,
            self.allow_empty_sets,
            num_threads,
            seed,
        )

        return y_set_pred

    def predict(self, X, alpha=None, binary_output=False, num_threads=4):
        """Predict class labels and prediction sets for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        alpha : float or None, default=None
            Desired miscoverage rate. The prediction sets will be constructed
            to contains the true label with probability at least 1-alpha. If None,
            only returns y_pred.

        binary_output : bool, default=False
            If True, returns prediction sets as binary arrays where 1 indicates
            the class is in the set. If False, returns lists of class labels.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels (point predictions).

        y_set_pred : list of arrays or ndarray of shape (n_samples, n_classes)
            If binary_output=False, returns a list where each element contains
            the classes in the prediction set for that sample.
            If binary_output=True, returns a binary array of shape
            (n_samples, n_classes) where 1 indicates class membership in the
            set.
        """

        if alpha is None:
            alpha = self.alpha_default

        if self.method in ["cv", "bootstrap"]:
            y_out = self._predict_cv(X, alpha, binary_output, num_threads)
        else:
            y_out = self._predict_split(X, alpha, binary_output)

        return y_out

    def _predict_cv(self, X, alpha=None, binary_output=False, num_threads=4):
        """Predict using CV+ or Jackknife+-after-Bootstrap method.

        This method will be called during prediction when `method='cv'`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        alpha : float or None, default=None
            Desired miscoverage rate. The prediction sets will be constructed
            to contains the true label with probability at least 1-alpha. If None,
            only returns y_pred.

        binary_output : bool, default=False
            If True, returns prediction sets as binary arrays where 1 indicates
            the class is in the set. If False, returns lists of class labels.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels (point predictions).

        y_set_pred : list of arrays or ndarray of shape (n_samples, n_classes)
            If binary_output=False, returns a list where each element contains
            the classes in the prediction set for that sample.
            If binary_output=True, returns a binary array of shape
            (n_samples, n_classes) where 1 indicates
            class membership in the set.
        """

        check_is_fitted(self)
        X = self._validate_X_predict(X)

        y_pred, oob_pred = self._compute_oob_predictions(X)
        y_pred = self.classes_.take(np.argmax(y_pred, axis=1), axis=0)
        if alpha is None:
            return y_pred
        else:
            test_giqs = self._compute_test_giqs(oob_pred, num_threads)
            compare_giqs = (self.train_giqs_[:, None, None] < test_giqs).sum(axis=0)
            compare_giqs = compare_giqs < ((1 - alpha) * (self._n_samples + 1))

            if binary_output:
                y_set_pred = compare_giqs.astype(int)
            else:
                y_set_pred = [
                    self.classes_.take(pred.nonzero()[0]) for pred in compare_giqs
                ]

            return y_pred, y_set_pred

    def _predict_split(self, X, alpha=None, binary_output=False, num_threads=4):
        """Predict using split conformal method.

        This method will be called during prediction when `method='split'`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        alpha : float or None, default=None
            Desired miscoverage rate. The prediction sets will be constructed
            to contains the true label with probability at least 1-alpha. If None,
            only returns y_pred.

        binary_output : bool, default=False
            If True, returns prediction sets as binary arrays where 1 indicates
            the class is in the set. If False, returns lists of class labels.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels (point predictions).

        y_set_pred : list of arrays or ndarray of shape (n_samples, n_classes)
            If binary_output=False, returns a list where each element contains
            the classes in the prediction set for that sample.
            If binary_output=True, returns a binary array of shape
            (n_samples, n_classes) where 1 indicates class membership in the
            set.
        """

        check_is_fitted(self)

        y_pred_proba = self.predict_proba(X)
        y_pred = self.classes_.take(np.argmax(y_pred_proba, axis=1), axis=0)

        if alpha is None:
            return y_pred
        else:
            tau = np.quantile(self.train_giqs_, 1 - alpha, method="higher")
            y_set_pred = self._predict_from_giqs(y_pred_proba, tau, num_threads)
            if not binary_output:
                y_set_pred = [
                    self.classes_.take(pred.nonzero()[0]) for pred in y_set_pred
                ]

            return y_pred, y_set_pred


class ConformalForestRegressor(
    ConformalRegressorMixin, BaseConformalForest, ForestRegressor
):
    """
    Base class for forest of conformal trees-based regressors.

    This class extends scikit-learn's ForestRegressor to implement conformal
    prediction.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        n_estimators=100,
        method="cv",
        cv=5,
        alpha_default=None,
        n_forests_per_fold=1,
        resample_n_estimators=True,
        estimator_params=tuple(),
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super().__init__(
            estimator=estimator,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            method=method,
            cv=cv,
            n_forests_per_fold=n_forests_per_fold,
            resample_n_estimators=resample_n_estimators,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.alpha_default = alpha_default

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, alpha=0.05, calib_size=0.3, sample_weight=None):
        """Fit the conformal forest regressor.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        calib_size : float, default=0.3
            Used when method='split'. The proportion of training samples to use
            for calibration.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted regressor.
        """

        self._fit_wrapper(X, y, calib_size, sample_weight)
        self.residuals_ = np.abs(self.y_.reshape(-1, 1) - self.oob_pred_.reshape(-1, 1))

        return self

    def predict(self, X, alpha=None, num_threads=4):
        """Predict regression values and prediction intervals for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        alpha : float or None, default=None
            Desired error rate. The prediction intervals will be constructed to
            contain the true value with probability at least 1-alpha. If None,
            only returns y_pred.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted regression values (point predictions).

        y_intervals : ndarray of shape (n_samples, 2)
            Prediction intervals for each sample. First column contains lower
            bounds, second column contains upper bounds.
        """

        if alpha is None:
            alpha = self.alpha_default

        if self.method in ["cv", "bootstrap"]:
            y_out = self._predict_cv(X, alpha, num_threads)
        else:
            y_out = self._predict_split(X, alpha)

        return y_out

    def _predict_cv(self, X, alpha=None, num_threads=4):
        """Predict using CV+ or Jackknife+-after-Bootstrap method.

        This method will be called during prediction when `method='cv'`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        alpha : float or None, default=None
            Desired error rate. The prediction intervals will be constructed to
            contain the true value with probability at least 1-alpha. If None,
            only returns y_pred.

        num_threads : int, default=4
            Number of threads to use for parallel computation.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted regression values (point predictions).

        y_intervals : ndarray of shape (n_samples, 2)
            Prediction intervals [lower, upper] for each sample computed using
            out-of-bag residuals.
        """

        check_is_fitted(self)
        X = self._validate_X_predict(X)

        y_pred, oob_pred = self._compute_oob_predictions(X)

        if alpha is None:
            return y_pred
        else:
            q_lo = np.quantile(
                oob_pred - self.residuals_, alpha, method="lower", axis=0
            )
            q_hi = np.quantile(
                oob_pred + self.residuals_, 1 - alpha, method="higher", axis=0
            )
            return y_pred, np.column_stack([q_lo, q_hi])

    def _predict_split(self, X, alpha=None):
        """Predict using split conformal method.

        This method will be called during prediction when `method='split'`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to predict.

        alpha : float or None, default=None
            Desired error rate. The prediction intervals will be constructed to
            contain the true value with probability at least 1-alpha. If None,
            only returns y_pred.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted regression values (point predictions).

        y_intervals : ndarray of shape (n_samples, 2)
            Prediction intervals [lower, upper] for each sample computed using
            calibration set residuals.
        """

        check_is_fitted(self)
        X = self._validate_X_predict(X)

        y_pred = FastRandomForestRegressor.predict(self, X)
        if alpha is None:
            return y_pred
        else:
            q = np.quantile(self.residuals_[:, 0], 1 - alpha, method="higher")
            return y_pred, np.column_stack([y_pred - q, y_pred + q])


class CoverForestClassifier(ConformalForestClassifier):
    """A conformal random forest classifier.

    This class provides an implementation of conformal random forest for
    prediction sets that contain the true labels with probability above 1-alpha,
    where alpha is a user-specified miscoverage rate. The prediction sets are
    constructed using the Adaptive Prediction Set (APS) method.

    The class supports three subsampling methods for out-of-sample calibration:

    - 'cv': Uses K-fold cross-validation to split the training set. This method
      is referred to as CV+.
    - 'bootstrap': Uses bootstrap subsampling on the training set. This method
      is referred to as Jackknife+-after-Bootstrap.
    - 'split': Uses train-test split on the training set. This method
      is referred to as split conformal.

    If there a lot of empty sets returned by the `predict()` method, try increasing
    the target coverage rate by decreasing the value of `alpha`. The option
    `allow_empty_sets=False` should be used sparingly.

    The Jackknife+-after-bootstrap implementation (`method='bootstrap'`) follows [5]
    Specifically, before fitting, the number of sub-estimators is resampled from the
    binomial distribution: Binomial(n_estimators / p, p) where

        p = 1 / (1 - n_samples)**max_samples.

    To fit the model with exactly `n_estimators` number of sub-estimators, initiate
    the model with `resample_n_estimators=False`.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of `sklearn.tree.DecisionTreeClassifier` in the forest.

    method : {'cv', 'bootstrap', 'split'}, default='cv'
        The conformal prediction method to use:

        - 'cv': Uses CV+ for conformal prediction
        - 'bootstrap': Uses Jackknife+-after-Bootstrap
        - 'split': Uses split conformal prediction

    cv : int or cross-validation generator, default=5
        Used when `method='cv'`. If an integer is provided, then it is the
        number of folds used. See the module sklearn.model_selection module for
        the list of possible cross-validation objects.

    k_init : int or "auto", default="auto"
        Initial value for the parameter k that penalizes any set prediction
        that contains more than k classes.
        If "auto", the value is chosen automatically during fitting.

    lambda_init : float or "auto", default="auto"
        Initial value for lambda parameter (regularization strength).
        If "auto", the value is chosen automatically during fitting.

    repeat_params_search : bool, default=True
        Whether to repeat the search for optimal parameters when refitting.

    allow_empty_sets : bool, default=True
        If True, allows empty prediction sets when no class meets the
        confidence threshold.

    randomized : bool, default=True
        If True, adds randomization during the label selection which yields
        smaller prediction sets. If False, the predictions will have more
        conservative coverage.

    alpha_default : float, default=None
        The default value of miscoverage rate `alpha` that will be passed to
        `predict()` whenever it is called indirectly i.e. via scikit-learn's
        `GridSearchCV`.

    n_forests_per_fold : int, default=1
        Used when `method='cv'`. The number of the forests to be fitted on each
        combination of K-1 folds.

    resample_n_estimators : bool, default=True
        Used when `method='bootstrap'`. If True, resample the value of
        `n_estimators` following the procedure in Kim, Xu & Barber (2020).
        Specifically, a new number of estimators is sampled from
        Binomial(n_estimators / p, p) where
        p = 1 / (1 - n_samples)**max_samples.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
        When `method='cv'`, the value will be passed on to the
        `FastRandomForestClassifier` sub-estimators.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(round(n_samples * max_samples), 1)` samples.

        Thus, `max_samples` should be in the interval `(0.0, 1.0]`.
        When `method='cv'`, the value will be passed on to the
        `FastRandomForestClassifier` sub-estimators.

    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see `tree_mathematical_formulation`.
        Note: This parameter is tree-specific.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, `sklearn.metrics.accuracy_score` is used.
        Provide a callable with signature `metric(y_true, y_pred)` to use a
        custom metric. Only available if `bootstrap=True`.

    n_jobs : int, default=None
        The number of jobs to run in parallel. `fit`, `predict`,
        `decision_path` and `apply` are all parallelized over the trees.
        ``None`` means 1 unless in a `joblib.parallel_backend` context.
        ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.

          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for classifications trained
        on data with missing values.

        The constraints hold over the probability of the positive class.

    Attributes
    ----------
    estimator_ : `FastRandomForestClassifier` or `sklearn.tree.DecisionTreeClassifier`
        The child estimator template used to create the collection of fitted
        sub-estimators. It will be a `FastRandomForestClassifier` if `method='cv'`
        and `sklearn.tree.DecisionTreeClassifier` otherwise.

    estimators_ : list of `FastRandomForestClassifier` or \
    `sklearn.tree.DecisionTreeClassifier`
        The collection of fitted sub-estimators. A list of
        `FastRandomForestClassifier` if `method='cv'` and a list of
        `sklearn.tree.DecisionTreeClassifier` otherwise.

    k_star_ : int
        The optimal k parameter found during fitting.

    lambda_star_ : float
        The optimal lambda parameter found during fitting.

    oob_pred_ : ndarray of shape (n_samples, n_classes, 1)
        The out-of-bag probability predictions on the training set.

    train_giqs_ : ndarray of shape (n_samples, n_classes)
        The generalized inverse quantile scores of the training set.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values).

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes) or \
            (n_samples, n_classes, n_outputs)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    See Also
    --------
    CoverForestRegressor : A conformal random forest for regression tasks.
    sklearn.ensemble.RandomForestClassifier : The standard random forest
        classifier from scikit-learn.

    References
    ----------
    .. [1] Leo Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    .. [2] Vladimir Vovk, Ilia Nouretdinov, Valery Manokhin & Alexander
           Gammerman, "Cross-conformal predictive distributions", 37-51,
           COPA 2018.
    .. [3] Yaniv Romano, Matteo Sesia & Emmanuel J. Candès, "Classification with
           Valid and Adaptive Coverage", NeurIPS 2020.
    .. [4] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael I. Jordan &
           Jitendra Malik, "Uncertainty Sets for Image Classifiers using
           Conformal Prediction", ICLR 2021.
    .. [5] Byol Kim, Chen Xu & Rina Foygel Barber, "Predictive inference is free
           with the jackknife+-after-bootstrap", NeurIPS 2020.

    Examples
    --------
    >>> from coverforest import CoverForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=4,
    ...                           n_informative=2, n_redundant=0,
    ...                           random_state=0, shuffle=False)
    >>> clf = CoverForestClassifier(n_estimators=10, method='cv', random_state=0)
    >>> clf.fit(X, y)
    CoverForestClassifier(...)
    >>> print(clf.predict(X[:1]))
    (array([0]), [array([0, 1])])
    """

    _parameter_constraints = {
        **RandomForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "method": [StrOptions({"bootstrap", "cv", "split"})],
        "cv": [Interval(Integral, 2, None, closed="left"), "cv_object"],
        "k_init": [StrOptions({"auto"}), Interval(Integral, 0, None, closed="left")],
        "lambda_init": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="right"),
            Interval(Integral, 0, None, closed="left"),
        ],
        "repeat_params_search": ["boolean"],
        "allow_empty_sets": ["boolean"],
        "randomized": ["boolean"],
        "alpha_default": [None, Interval(Real, 0, None, closed="right")],
        "resample_n_estimators": ["boolean"],
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=5,
        *,
        method="cv",
        cv=5,
        k_init="auto",
        lambda_init="auto",
        repeat_params_search=True,
        allow_empty_sets=True,
        randomized=True,
        alpha_default=None,
        n_forests_per_fold=1,
        resample_n_estimators=True,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        max_samples=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        estimator_params = (
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "random_state",
            "ccp_alpha",
            "monotonic_cst",
        )

        if isinstance(method, str) and method == "cv":
            estimator = FastRandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                warm_start=warm_start,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
            )
            estimator_params += (
                "bootstrap",
                "max_samples",
                "n_jobs",
                "oob_score",
                "warm_start",
            )
        else:
            estimator = DecisionTreeClassifier()

        super().__init__(
            estimator=estimator,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            method=method,
            cv=cv,
            k_init=k_init,
            lambda_init=lambda_init,
            repeat_params_search=repeat_params_search,
            allow_empty_sets=allow_empty_sets,
            randomized=randomized,
            alpha_default=alpha_default,
            n_forests_per_fold=n_forests_per_fold,
            resample_n_estimators=resample_n_estimators,
            max_samples=max_samples,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
        )

        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha


class CoverForestRegressor(ConformalForestRegressor):
    """A conformal random forest regressor.

    This class provides an implementation of conformal random forest for
    prediction intervals that contain the true target value with probability
    above 1-alpha, where alpha is a user-specified error rate.

    The class supports three subsampling methods for out-of-sample calibration:

    - 'cv': Uses K-fold cross-validation to split the training set. This method
      is referred to as CV+.
    - 'bootstrap': Uses bootstrap subsampling on the training set. This method
      is referred to as Jackknife+-after-Bootstrap.
    - 'split': Uses train-test split on the training set. This method
      is referred to as split conformal.

    The Jackknife+-after-bootstrap implementation (`method='bootstrap'`) follows [5]
    Specifically, before fitting, the number of sub-estimators is resampled from the
    binomial distribution: Binomial(n_estimators / p, p) where

        p = 1 / (1 - n_samples)**max_samples.

    To fit the model with exactly `n_estimators` number of sub-estimators, initiate
    the model with `resample_n_estimators=False`.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of `sklearn.tree.DecisionTreeClassifier` in the forest.

    method : {'cv', 'bootstrap', 'split'}, default='cv'
        The conformal prediction method to use:

        - 'cv': Uses CV+ for conformal prediction
        - 'bootstrap': Uses Jackknife+-after-Bootstrap
        - 'split': Uses split conformal prediction

    cv : int or cross-validation generator, default=5
        Used when `method='cv'`. If an integer is provided, then it is the
        number of folds used. See the module sklearn.model_selection module for
        the list of possible cross-validation objects.

    alpha_default : float, default=None
        The default value of miscoverage rate `alpha` that will be passed to
        `predict()` whenever it is called indirectly i.e. via scikit-learn's
        `GridSearchCV`.

    n_forests_per_fold : int, default=1
        Used when `method='cv'`. The number of the forests to be fitted on each
        combination of K-1 folds.

    resample_n_estimators : bool, default=True
        Used when `method='bootstrap'`. If True, resample the value of
        `n_estimators` following the procedure in Kim, Xu & Barber (2020).
        Specifically, a new number of estimators is sampled from
        Binomial(n_estimators / p, p) where
        p = 1 / (1 - n_samples)**max_samples.

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum number of
          samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        .. note::
            The default of 1.0 is equivalent to bagged trees and more
            randomness can be achieved by setting smaller values, e.g. 0.3.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(round(n_samples * max_samples), 1)` samples.

        Thus, `max_samples` should be in the interval `(0.0, 1.0]`.
        When `method='cv'`, the value will be passed on to the
        `FastRandomForestClassifier` sub-estimators.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
        When `method='cv'`, the value will be passed on to the
        `FastRandomForestClassifier` sub-estimators.

    oob_score : bool or callable, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        By default, `sklearn.metrics.r2_score` is used.
        Provide a callable with signature `metric(y_true, y_pred)` to use a
        custom metric. Only available if `bootstrap=True`.

    n_jobs : int, default=None
        The number of jobs to run in parallel. `fit`, `predict`,
        `decision_path` and `apply` are all parallelized over the trees.
        ``None`` means 1 unless in a `joblib.parallel_backend` context.
        ``-1`` means using all processors.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.

          - 1: monotonically increasing
          - 0: no constraint
          - -1: monotonically decreasing

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for regressions trained on
        data with missing values.

    Attributes
    ----------
    estimator_ : `FastRandomForestClassifier` or `sklearn.tree.DecisionTreeRegressor`
        The child estimator template used to create the collection of fitted
        sub-estimators. It will be a `FastRandomForestClassifier` if `method='cv'`
        and `sklearn.tree.DecisionTreeRegressor` otherwise.

    estimators_ : list of `FastRandomForestRegressor` or \
    `sklearn.tree.DecisionTreeRegressor`
        The collection of fitted sub-estimators. A list of
        `FastRandomForestClassifier` if `method='cv'` and a list of
        `sklearn.tree.DecisionTreeRegressor` otherwise.

    oob_pred_ : ndarray of shape (n_samples,)
        The out-of-bag predictions on the training data.

    residuals_ : ndarray of shape (n_samples,)
        The out-of-bag residuals on the training data.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    See Also
    --------
    CoverForestClassifier : A conformal random forest for classification tasks.
    sklearn.ensemble.RandomForestRegressor : The standard random forest
        regressor from scikit-learn.

    Notes
    -----
    The conformal prediction with K-fold cross-validation (CV+) method was
    proposed by Romano, Sesia & Candès (2020). The conformal prediction with
    bootstrap subsampling (Jackknife+-after-Bootstrap) was proposed by Kim, Xu
    & Barber (2020).

    References
    ----------
    .. [1] Leo Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    .. [2] Vladimir Vovk, Ilia Nouretdinov, Valery Manokhin & Alexander
           Gammerman, "Cross-conformal predictive distributions", 37-51,
           COPA 2018.
    .. [3] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas &
           Ryan J. Tibshirani, "Predictive inference with the jackknife+",
           Ann. Statist. 49 (1) 486-507, 2021.
    .. [4] Byol Kim, Chen Xu, Rina Foygel Barber, "Predictive inference is free
           with the jackknife+-after-bootstrap", NeurIPS 2020.

    Examples
    --------
    >>> from coverforest import CoverForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = CoverForestRegressor(n_estimators=10, method='cv', random_state=0)
    >>> regr.fit(X, y)
    CoverForestRegressor(...)
    >>> print(regr.predict([[0, 0, 0, 0]]))
    (array([13.14530751]), array([[-44.87496872,  73.12034704]]))
    """

    _parameter_constraints = {
        **RandomForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
        "method": [StrOptions({"bootstrap", "cv", "split"})],
        "cv": [Interval(Integral, 2, None, closed="left"), "cv_object"],
        "alpha_default": [None, Interval(Real, 0, None, closed="right")],
        "resample_n_estimators": ["boolean"],
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=5,
        *,
        method="cv",
        cv=5,
        alpha_default=None,
        n_forests_per_fold=1,
        resample_n_estimators=True,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        max_samples=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        estimator_params = (
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "random_state",
            "ccp_alpha",
            "monotonic_cst",
        )

        if isinstance(method, str) and method == "cv":
            estimator = FastRandomForestRegressor(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                ccp_alpha=ccp_alpha,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                warm_start=warm_start,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
            )
            estimator_params += (
                "bootstrap",
                "max_samples",
                "n_jobs",
                "oob_score",
                "warm_start",
            )
        else:
            estimator = DecisionTreeRegressor()

        super().__init__(
            estimator=estimator,
            estimator_params=estimator_params,
            n_estimators=n_estimators,
            method=method,
            cv=cv,
            alpha_default=alpha_default,
            n_forests_per_fold=n_forests_per_fold,
            resample_n_estimators=resample_n_estimators,
            max_samples=max_samples,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

        self.bootstrap = bootstrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha
