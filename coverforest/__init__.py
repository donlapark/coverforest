"""Random forests with conformal methods for set prediction and interval prediction."""

# Authors: Donlapark Ponnoprat <donlapark@gmail.com>
#          Panisara Meehinkong <Panisara.nf@gmail.com>
# License: BSD 3 clause

from ._forest import (
    CoverForestClassifier,
    CoverForestRegressor,
)

__all__ = [
    "CoverForestClassifier",
    "CoverForestRegressor",
]
