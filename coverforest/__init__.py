"""Random forests with conformal methods for set prediction and interval prediction."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from ._forest import (
    CoverForestClassifier,
    CoverForestRegressor,
)

__all__ = [
    "CoverForestClassifier",
    "CoverForestRegressor",
]
