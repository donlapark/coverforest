"""This file shows how to write test based on the scikit-learn common tests."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from sklearn.utils.estimator_checks import parametrize_with_checks

from coverforest import CoverForestClassifier, CoverForestRegressor


def report_failed_check_bootstrap(estimator):
    if hasattr(estimator, "bootstrap"):
        if estimator.bootstrap:
            return {
                "check_sample_weight_equivalence_on_dense_data": (
                    "Known issue of bootstrap subsampling"
                    "incorrectly handling sample weights."
                ),
                "check_sample_weight_equivalence_on_sparse_data": (
                    "Known issue of bootstrap subsampling"
                    "incorrectly handling sample weights."
                ),
            }
    return {}


# parametrize_with_checks allows to get a generator of check that is more fine-grained
# than check_estimator
@parametrize_with_checks(
    [
        CoverForestClassifier(method="cv"),
        CoverForestClassifier(method="bootstrap"),
        CoverForestClassifier(method="split"),
        CoverForestRegressor(method="cv"),
        CoverForestRegressor(method="bootstrap"),
        CoverForestRegressor(method="split"),
    ],
    expected_failed_checks=report_failed_check_bootstrap,
)
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
