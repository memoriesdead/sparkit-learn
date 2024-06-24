# encoding: utf-8

import numpy as np
import scipy.sparse as sp
import sklearn
from pyspark.mllib.clustering import KMeans as MLlibKMeans
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin
from pyspark.rdd import RDD
from typing import Type, Union
import matplotlib.pyplot as plt

class SparkKMeans(KMeans, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, Z: Union[RDD, np.ndarray, sp.spmatrix], random_state: int = None) -> Type['SparkKMeans']:
        """Compute k-means clustering.

        Parameters
        ----------
        Z : Union[RDD, np.ndarray, sp.spmatrix]
            Train data.
        random_state : int, optional
            Random seed used for initial centroid selection.

        Returns
        -------
        self
        """
        # Set random seed if specified
        if random_state is not None:
            self.random_state = random_state

        if isinstance(Z, RDD):
            X = Z[:, 'X']
        else:
            X = Z

        if self.init == 'k-means||':
            self._mllib_model = MLlibKMeans.train(
                X.unblock(),
                self.n_clusters,
                maxIterations=self.max_iter,
                initializationMode="k-means||")
            self.cluster_centers_ = self._mllib_model.centers
        else:
            models = X.map(lambda X: super().fit(X))
            models = models.map(lambda model: model.cluster_centers_).collect()
            return super().fit(np.concatenate(models))

    def predict(self, X: Union[RDD, np.ndarray, sp.spmatrix]) -> RDD:
        if hasattr(self, '_mllib_model'):
            if isinstance(X, RDD):
                X = X.unblock()
            return X.map(lambda x: self._mllib_model.predict(x))
        else:
            return X.map(lambda X: super().predict(X))

    def transform(self, X: Union[RDD, np.ndarray, sp.spmatrix]) -> RDD:
        """Assign each point in X to the closest cluster.

        Parameters
        ----------
        X : Union[RDD, np.ndarray, sp.spmatrix]
            Data to transform.

        Returns
        -------
        RDD
            Cluster assignments for each point in X.
        """
        return self.predict(X)

    def _calculate_wcss(self, X: Union[RDD, np.ndarray, sp.spmatrix], assignments: RDD) -> float:
        """Calculate the within-cluster sum of squares (WCSS).

        Parameters
        ----------
        X : Union[RDD, np.ndarray, sp.spmatrix]
            Training data.
        assignments : RDD
            Cluster assignments for each point in X.

        Returns
        -------
        float
            WCSS.
        """
        wcss = 0.0
        for i, center in enumerate(self.cluster_centers_):
            cluster_points = X[assignments == i]
            for point in cluster_points:
                wcss += np.sum((point - center) ** 2)
        return wcss

    def plot_clusters(self, X: Union[RDD, np.ndarray, sp.spmatrix], assignments: RDD):
        """Plot the clusters.

        Parameters
        ----------
        X : Union[RDD, np.ndarray, sp.spmatrix]
            Training data.
        assignments : RDD
            Cluster assignments for each point in X.
        """
        # Extract coordinates for each point
        if isinstance(X, RDD):
            X = X.unblock()
        x_coordinates = X[:, 0]
        y_coordinates = X[:, 1]

        # Create a scatter plot with different colors for each cluster
        plt.scatter(x_coordinates, y_coordinates, c=assignments)
        plt.show()

