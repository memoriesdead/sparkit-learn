import numpy as np
from typing import List, Tuple
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class SparkRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Distributed random forest classifier for Spark.

    This classifier wraps the Spark MLLib implementation of random forests
    and exposes a scikit-learn-compatible API.

    Parameters
    ----------
    n_estimators : int, optional (default=10)
        Number of trees in the forest.
    criterion : {'gini', 'entropy'}, optional (default='gini')
        The function to measure the quality of a split.
    max_depth : int, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.
    max_features : int, optional (default=None)
        The number of features to consider when looking forthe best split. If None, then all features are considered.
bootstrap : bool, optional (default=True)
Whether or not to use bootstrapping when sampling the training data.
oob_score : bool, optional (default=False)
Whether or not to use out-of-bag samples to estimate the generalization
error.
n_jobs : int, optional (default=1)
The number of jobs to run in parallel for both fit and predict.
random_state : int or RandomState, optional (default=None)
Seed for the random number generator.
sc : SparkContext, optional (default=None)
The SparkContext to use for training and prediction. If None, then a
new SparkContext is created using the default configuration.
"""
def init(self, n_estimators: int = 10, criterion: str = 'gini',
max_depth: int = None, min_samples_split: int = 2,
min_samples_leaf: int = 1, max_features: int = None,
bootstrap: bool = True, oob_score: bool = False,
n_jobs: int = 1, random_state: int = None,
sc: SparkContext = None):
self.n_estimators = n_estimators
self.criterion = criterion
self.max_depth = max_depth
self.min_samples_split = min_samples_split
self.min_samples_leaf = min_samples_leaf
self.max_features = max_features
self.bootstrap = bootstrap
self.oob_score = oob_score
self.n_jobs = n_jobs
self.random_state = random_state
self.sc = sc or SparkContext.getOrCreate()


