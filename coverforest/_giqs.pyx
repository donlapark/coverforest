# distutils: language=c++
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3

from libc.math cimport isnan, floor
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy, memset


from libcpp.algorithm cimport sort
from libcpp.pair cimport pair
from libcpp.set cimport set
from libcpp.vector cimport vector

cimport cython
from cython.parallel cimport prange, parallel

import numpy as np
cimport numpy as cnp

cimport openmp

from sklearn.utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t, int32_t, int64_t
# Note: _tree uses cimport numpy, cnp.import_array, so we need to include
# numpy headers, see setup.py.
from sklearn.tree._tree cimport Node
from sklearn.tree._tree cimport Tree
from sklearn.tree._utils cimport safe_realloc


cnp.import_array()

cdef intp_t TREE_LEAF = -1


cdef extern from "omp.h" nogil:
    int omp_get_thread_num()
    int omp_get_max_threads()


cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)

cdef extern from *:
    """
    #include <algorithm>
    #include <vector>

    void argsort(int32_t* indices, const double* scores, size_t n) {
        std::vector<int32_t> idx(n);
        for (size_t i = 0; i < n; ++i) {
            idx[i] = static_cast<int32_t>(i);
        }

        std::sort(idx.begin(), idx.end(),
                 [scores](int32_t i1, int32_t i2) {
                     return scores[i1] > scores[i2];
                 });

        // Copy sorted indices back to output array
        for (size_t i = 0; i < n; ++i) {
            indices[i] = idx[i];
        }
    }
    """
    void argsort(int32_t* indices, const double* scores, size_t n) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _argsort(int32_t* indices_ptr, float64_t* scores_ptr, size_t n) noexcept nogil:
    argsort(indices_ptr, scores_ptr, n)


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_test_giqs_cv(const float64_t[:,:,::1] oob_pred,
                        float64_t[:,:,::1] out,
                        int64_t k_star,
                        float32_t lambda_star,
                        bint randomized,
                        bint allow_empty_sets,
                        intp_t num_threads,
                        intp_t random_state):
    """Compute generalized inverse quantiles for CV+ and Jackknife+-after-Bootstrap methods.

    Compute the regularized generalized inverse quantile comformity
    scores (giqs) of test samples.

    This method will be called during prediction when `method='cv'`.

    Given a test sample with its out-of-bag probability predictions ordered
    from the most likely to the least likely classes π̂₁ ≥ ... ≥ π̂ₗ, the
    sample's (non-regularized) giqs for a label j is obtained by first
    sampling U from Uniform[0, 1] and then calculate::

        π̂₁ + ... + π̂ⱼ₋₁ + U * π̂ⱼ.

    The regularized giqs is then obtained by adding a term that penalizes
    prediction sets that are larger than k::

        π̂₁ + ... + π̂ⱼ₋₁ + U * π̂ⱼ + λ * max(0, j - k + 1).

    This method will be called during prediction when `method='cv'` or
    `method='bootstrap'`.

    Parameters
    ----------
    oob_pred : memoryview, shape (n_train, n_classes, n_test)
        Out-of-bag prediction probabilities from the ensemble model.
        Each element [i,j,k] represents the probability of test sample k
        belonging to class j as predicted by the ensemble of all sub-estimators
       that did not see sample i during training.

    out : memoryview, shape (n_train, n_classes, n_test)
        Pre-allocated array to store the computed GIQ scores.
        Will be modified in-place.

    k_star : int
        Optimal value for the parameter k that penalizes prediction sets
        containing more than k classes.

    lambda_star : float
        Optimal regularization strength parameter that controls the
        trade-off between set size and coverage.

    randomized : bool
        If True, adds randomization during label selection to produce
        smaller prediction sets. If False, predictions will have more
        conservative coverage.

    allow_empty_sets : bool
        If True, allows empty prediction sets when no class meets the
        confidence threshold.

    num_threads : int
        Number of threads to use for parallel computation.

    random_state : int
        The random seed for sampling U from the uniform distribution U[0,1].

    References
    ----------
    .. [1] Yaniv Romano, Matteo Sesia & Emmanuel J. Candès, "Classification
           with Valid and Adaptive Coverage", NeurIPS 2020.
    .. [2] Anastasios Nikolas Angelopoulos, Stephen Bates,
           Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
           Classifiers using Conformal Prediction", ICLR 2021.
    """

    cdef Py_ssize_t i, j, k, l
    cdef Py_ssize_t n_samples = oob_pred.shape[0]
    cdef Py_ssize_t n_test = oob_pred.shape[1]
    cdef Py_ssize_t n_classes = oob_pred.shape[2]
    cdef int64_t samples_stride = n_test * n_classes
    cdef int64_t ij_idx
    cdef int32_t* I
    cdef float64_t* sorted_scores
    cdef float64_t* penalty
    cdef float64_t* E
    cdef float32_t* U = <float32_t*>malloc(n_test * sizeof(float32_t))
    cdef mt19937 gen
    cdef uniform_real_distribution[float32_t] dist
    cdef float64_t* temp
    cdef int32_t* indices

    if k_star > n_classes:
        k_star = n_classes
    if k_star < 0:
        k_star = 0

    try:
        with nogil, parallel(num_threads=num_threads):
            indices = <int32_t*>malloc(n_classes * sizeof(int32_t))
            temp = <float64_t*>malloc(n_classes * sizeof(float64_t))
            I = <int32_t*>malloc(n_samples * n_test * n_classes * sizeof(int32_t))
            sorted_scores = <float64_t*>malloc(n_samples * n_test * n_classes * sizeof(float64_t))
            for i in prange(n_samples):
                for j in range(n_test):
                    ij_idx = i * samples_stride + j * n_classes
                    memcpy(temp, &oob_pred[i, j, 0], n_classes * sizeof(float64_t))
                    _argsort(indices, temp, n_classes)
                    for k in range(n_classes):
                        I[ij_idx + k] = indices[k]
                        sorted_scores[ij_idx + k] = oob_pred[i, j, indices[k]]

                    out[i, j, 0] = sorted_scores[ij_idx]
                    for k in range(1, n_classes):
                        out[i, j, k] = out[i, j, k - 1] + sorted_scores[ij_idx + k]

            free(temp)
            free(indices)

            if not randomized:
                for i in prange(n_samples):
                    for j in range(n_test):
                        for k in range(k_star, n_classes):
                            out[i, j, k] += lambda_star

            else:
                penalty = <float64_t*>malloc(n_classes * sizeof(float64_t))
                E = <float64_t*>malloc(n_samples * n_test * n_classes * sizeof(float64_t))
                dist = uniform_real_distribution[float32_t](0.0, 1.0)
                gen = mt19937(random_state)
                for l in prange(n_test):
                    U[l] = dist(gen)

                memset(penalty, 0, n_classes * sizeof(float64_t))
                for k in prange(k_star, n_classes):
                    penalty[k] = lambda_star * (k - k_star + 1)

                for i in prange(n_samples):
                    for j in range(n_test):
                        ij_idx = i * samples_stride + j * n_classes
                        if allow_empty_sets:
                            E[ij_idx] = U[j] * out[i, j, 0] + penalty[0]
                        else:
                            E[ij_idx] = out[i, j, 0] + penalty[0]

                        for k in range(1, n_classes):
                            E[ij_idx + k] = (U[j] * sorted_scores[ij_idx + k] +
                                         out[i, j, k - 1] + penalty[k])

                        for k in range(n_classes):
                            out[i, j, I[ij_idx + k]] = E[ij_idx + k]

                free(I)
                free(E)
                free(penalty)
                free(sorted_scores)
    finally:
        free(U)


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_predictions_split(const float64_t[:,::1] oob_pred,
                                 float64_t[:,::1] out,
                                 float64_t tau,
                                 int64_t k_star,
                                 float32_t lambda_star,
                                 bint randomized,
                                 bint allow_empty_sets,
                                 intp_t num_threads,
                                 intp_t random_state):
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
    oob_pred : memoryview, shape (n_samples, n_classes)
        Prediction probabilities from the model.
        Each element [i,j] represents the out-of-bag predicted probability of
        sample i belonging to class j.

    out : memoryview, shape (n_samples, n_classes)
        Pre-allocated binary array to store the prediction set.
        out[i,j] = 1 indicates class j is in the prediction set.
        Will be modified in-place.

    tau : float
        Pre-computed threshold based on the calibration set, used to determine
        which classes to include in the prediction sets.

    k_star : int
        Optimal value for the parameter k that penalizes prediction sets
        containing more than k classes.

    lambda_star : float
        Optimal regularization strength parameter that controls the
        trade-off between set size and coverage.

    randomized : bool
        If True, adds randomization during label selection to produce
        smaller prediction sets. If False, predictions will have more
        conservative coverage.

    allow_empty_sets : bool
        If True, allows empty prediction sets when no class meets the
        confidence threshold.

    num_threads : int
        Number of threads to use for parallel computation.

    random_state : int
        The random seed for sampling U from the uniform distribution U[0,1].

    References
    ----------
    .. [1] Anastasios Nikolas Angelopoulos, Stephen Bates,
           Michael I. Jordan & Jitendra Malik, "Uncertainty Sets for Image
           Classifiers using Conformal Prediction", ICLR 2021.
    """

    cdef Py_ssize_t i, j, k, l, inc, ij_idx
    cdef Py_ssize_t n_samples = oob_pred.shape[0]
    cdef Py_ssize_t n_classes = oob_pred.shape[1]
    cdef int32_t* indices
    cdef float64_t* sorted_scores
    cdef float64_t* cumsum
    cdef float64_t* penalties
    cdef float64_t* penalties_cumsum
    cdef int32_t* sizes_base
    cdef int32_t* sizes
    cdef float64_t* V
    cdef float32_t* U = <float32_t*>malloc(n_samples * sizeof(float32_t))
    cdef mt19937 gen
    cdef uniform_real_distribution[float32_t] dist
    cdef float64_t numerator, denominator
    cdef float64_t* temp

    try:
        with nogil, parallel(num_threads=num_threads):
            indices = <int32_t*>malloc(n_samples * n_classes * sizeof(int32_t))
            sorted_scores = <float64_t*>malloc(n_samples * n_classes * sizeof(float64_t))
            cumsum = <float64_t*>malloc(n_samples * n_classes * sizeof(float64_t))
            penalties = <float64_t*>malloc(n_classes * sizeof(float64_t))
            penalties_cumsum = <float64_t*>malloc(n_classes * sizeof(float64_t))
            L = <int32_t*>malloc(n_samples * sizeof(int32_t))
            temp = <float64_t*>malloc(n_classes * sizeof(float64_t))

            for k in prange(n_classes):
                if k >= k_star:
                    penalties[k] = lambda_star
                else:
                    penalties[k] = 0

            memset(penalties_cumsum, 0, n_classes * sizeof(float64_t))
            for k in prange(k_star, n_classes):
                penalties_cumsum[k] = lambda_star * (k - k_star + 1)

            for i in prange(n_samples):
                inc = i * n_classes
                for j in range(n_classes):
                    temp[j] = oob_pred[i, j]

                _argsort(indices + inc, temp, n_classes)

                sorted_scores[inc] = oob_pred[i, indices[inc]]
                cumsum[inc] = sorted_scores[inc]
                for j in range(1, n_classes):
                    ij_idx = inc + j
                    sorted_scores[ij_idx] = oob_pred[i, indices[ij_idx]]
                    cumsum[ij_idx] = cumsum[ij_idx - 1] + sorted_scores[ij_idx]

                L[i] = 1
                for j in range(n_classes):
                    if (cumsum[inc + j] + penalties_cumsum[j]) <= tau:
                        L[i] = j + 2
                L[i] = min(L[i], n_classes)

            free(temp)
            free(penalties_cumsum)
            free(sorted_scores)
            free(cumsum)

            if randomized:
                V = <float64_t*>malloc(n_samples * sizeof(float64_t))

                dist = uniform_real_distribution[float32_t](0.0, 1.0)
                gen = mt19937(random_state)
                for l in prange(n_samples):
                    U[l] = dist(gen)

                for i in prange(n_samples):
                    inc = i * n_classes
                    j = L[i] - 1
                    numerator = tau
                    if j > 0:
                        numerator -= cumsum[inc + j - 1]
                    numerator = numerator - penalties_cumsum[j]
                    denominator = sorted_scores[inc + j] + penalties[j]

                    if denominator != 0:
                        V[i] = numerator / denominator
                    else:
                        V[i] = 0

                    L[i] = L[i] - (U[i] >= V[i])

                free(V)
                free(penalties)

            if tau == 1.0:
                for i in prange(n_samples):
                    L[i] = n_classes

            if not allow_empty_sets:
                for i in prange(n_samples):
                    if L[i] == 0:
                        L[i] = 1

            for i in prange(n_samples):
                inc = i * n_classes
                for j in range(L[i]):
                    out[i, indices[inc + j]] = 1

            free(indices)
            free(L)

    finally:
        free(U)
