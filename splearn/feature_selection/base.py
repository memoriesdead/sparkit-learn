# -*- coding: utf-8 -*-
"""Generic feature selection mixin"""

from abc import ABCMeta

from sklearn.externals import six
from sklearn.feature_selection.base import SelectorMixin

from ..base import SparkBroadcasterMixin, SparkTransformerMixin


class SparkSelectorMixin(six.with_metaclass(ABCMeta, SelectorMixin,
                                            SparkTransformerMixin,
                                            SparkBroadcasterMixin)):

    """
    Tranformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_support_mask`.
    """

    def __init__(self, max_features=None):
        self.max_features = max_features

    def fit(self, X, y=None):
        """Fit the feature selector to the training data

        Parameters
        ----------
        X : array-like or RDD
            Training data.
        y : array-like, optional
            Target labels.

        Returns
        -------
        self
        """
        self._fit(X, y)
        return self
