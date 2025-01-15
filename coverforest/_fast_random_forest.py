"""
Fast Random Forest prediction methods.

Modifications of Random Forest Classifier/Regressor to skip input checks, as these
checks are already parts of the fitting and prediction methods in CoverForestClassifier
and CoverForestRegressor.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Donlapark Ponnoprat <donlapark.p@cmu.ac.th>
#
# License: BSD 3 clause

import threading
from abc import abstractmethod
from numbers import Integral
from warnings import warn

import numpy as np
import sklearn
from scipy.sparse import issparse
from sklearn.base import _fit_context
from sklearn.ensemble._forest import (
    BaseForest,
    ForestClassifier,
    ForestRegressor,
    _accumulate_prediction,
    _get_n_samples_bootstrap,
    _partition_estimators,
)
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
    validate_data,
)

MAX_INT = np.iinfo(np.int32).max


class BaseFastForest(BaseForest):
    """
    This class extends sklearn's `BaseForest` to allow skipping input checks,
    as these checks are already parts of the fitting and prediction methods in
    CoveRFClassifier and CoveRFRegressor.

    Warning: This class should not be used directly. Use the derived classes instead.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "bootstrap": ["boolean"],
        "oob_score": ["boolean", callable],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "max_samples": [
            None,
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, check_input=False):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : bool, default=False
            Whether to perform input checks before fitting the estimator.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if check_input:
            # Validate or convert input data
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

            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X)

            if issparse(X):
                X.sort_indices()

            y = np.atleast_1d(y)
            if y.ndim == 2 and y.shape[1] == 1:
                warn(
                    (
                        "A column-vector y was passed when a 1d array was"
                        " expected. Please change the shape of y to "
                        "(n_samples,), for example using ravel()."
                    ),
                    DataConversionWarning,
                    stacklevel=2,
                )

            if y.ndim == 1:
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

            y, expanded_class_weight = self._validate_y_class_weight(y)

            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)

            if expanded_class_weight is not None:
                if sample_weight is not None:
                    sample_weight = sample_weight * expanded_class_weight
                else:
                    sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(sklearn.ensemble._forest._parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            self.estimators_.extend(trees)

        return self


class FastRandomForestClassifier(BaseFastForest, ForestClassifier):
    """
    This class extends sklearn's `RandomForestClassifier` to allow skipping
    input checks, as these checks are already parts of the fitting and
    prediction methods in CoveRFClassifier and CoveRFRegressor.

    Warning: This class should not be used directly. Use the main methods instead.
    """

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
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
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
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
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.class_weight = class_weight
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

    def predict_proba(self, X, check_input=False):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        check_input : bool, default=False
            Whether to perform input checks before fitting the estimator.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        if check_input:
            check_is_fitted(self)
            # Check data
            X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        sklearn.ensemble._forest.Parallel(
            n_jobs=n_jobs, verbose=self.verbose, require="sharedmem"
        )(
            delayed(_accumulate_prediction)(e.predict_proba, X, all_proba, lock)
            for e in self.estimators_
        )

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba


class FastRandomForestRegressor(BaseFastForest, ForestRegressor):
    """
    This class extends sklearn's `RandomForestRegressor` to allow skipping
    input checks, as these checks are already parts of the fitting and
    prediction methods in CoveRFClassifier and CoveRFRegressor.

    Warning: This class should not be used directly. Use the main methods instead.
    """

    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
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
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def predict(self, X, check_input=False):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        check_input : bool, default=False
            Whether to perform input checks before fitting the estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        if check_input:
            check_is_fitted(self)
            # Check data
            X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat
