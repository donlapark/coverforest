"""Random forests with conformal methods for set prediction and interval prediction."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

"""Ensemble-based methods for classification, regression and anomaly detection."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from ._forest import (
    CoverForestClassifier,
    CoverForestRegressor,
)

__all__ = [
    "CoverForestClassifier",
    "CoverForestRegressor",
]
