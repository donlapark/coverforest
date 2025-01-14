"""This file will just show how to write tests for the template classes."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from coverforest import CoverForestClassifier

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_classifier(data):
    """Check the internals and behaviour of `TemplateClassifier`."""
    X, y = data
    clf = CoverForestClassifier()

    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "y_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0], np.unique(y).shape[0])
