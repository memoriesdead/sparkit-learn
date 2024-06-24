import numpy as np
from typing import List, Tuple
from pyspark import SparkContext
from pyspark.mllib._common import (_get_unmangled_double_vector_rdd,
                                   _serialize_double_vector)

class DbscanModel:
    """Model for DBSCAN clustering.

    This model wraps the Spark MLLib implementation of DBSCAN and exposes a
    scikit-learn-compatible API.

    Parameters
    ----------
    model : pyspark.mllib.clustering.DbscanModel
        The trained DBSCAN model.
    """
    def __init__(self, model):
        self._model = model

    def predict(self, points: List[Tuple[float]]) -> List[int]:
        """Predict the cluster labels for the given points.

        Parameters
        ----------
        points : list of tuples of floats
            The points to classify. Each point is a tuple of floats.

        Returns
        -------
        labels : list of ints
            The cluster labels for each point.
        """
        serialized = [_serialize_double_vector(point) for point in points]
        return self._model.predict(serialized)

class Dbscan:
    """Distributed DBSCAN clustering for Spark.

    This class wraps the Spark MLLib implementation of DBSCAN and exposes a
    scikit-learn-compatible API.

    Parameters
    ----------
    epsilon : float
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other.
    num_points : int
        The number of points in a neighborhood for a point to be considered as
        a core point. This includes the point itself.
    sc : SparkContext, optional (default=None)
        The SparkContext to use for training and prediction. If None, then a
        new SparkContext is created using the default configuration.
    """
    def __init__(self, epsilon: float, num_points: int, sc: SparkContext = None):
        self.epsilon = epsilon
        self.num_points = num_points
        self.sc = sc

    def fit(self, points: List[Tuple[float]]) -> DbscanModel:
        """Fit the DBSCAN model to the given points.

        Parameters
        ----------
        points : list of tuples of floats
            The points to cluster. Each point is a tuple of floats.

        Returns
        -------
        model : DbscanModel
            The trained DBSCAN model.
        """
        sc = self.sc or SparkContext.getOrCreate()
        jrdd = _get_unmangled_double_vector_rdd(points)._jrdd
        model = sc._jvm.PythonDbscanAPI().train(jrdd, self.epsilon, self.num_points)
        return DbscanModel(model)

        
