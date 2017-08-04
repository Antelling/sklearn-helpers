from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math


class LOWESS(BaseEstimator, RegressorMixin):
    """
    Locally Weighted Scatterplot Smoothing

    LOWESS trains a custom model for each point of prediction, making it extremely computationally inefficient.

    Parameters
    ----------
    base_estimator : object, optional
        Base estimator object which implements the BaseEstimator and RegressorMixin APIs.
        Defaults to LinearRegression()

    weight : {'tri-cube', 'identity', 'normal''}, optional (default='tri-cube')
        The weight function that gives the most importance to data points closest to the point of estimation.

        * `tri-cube` is traditional for lowess, and is defined as (1 - np.abs(x)**3)**3
        * `range` uses a constant weight for every point within `range/2` distance from the point of estimation
        * `normal` uses a normal curve centered on the point of estimation with a std dev of `sigma`

    sigma : float, optional (default .1 * diameter of point cloud)
        Sigma of the normal weight function. Ignored by other weight functions.

    range : float, optional, (default .1 * diameter of point cloud)
        Range of the range weight function. Ignored by other weight functions.

    """

    def __init__(self, base_estimator=LinearRegression(), weight="tri-cube", divisor=10, a=None, diameter=None):
        self.base_estimator = base_estimator
        assert weight in ["tri-cube", "range", "normal"]
        self.weight = weight
        self.diameter = diameter
        self.divisor = divisor
        self.a = a

    def fit(self, X, y):
        # store X and y, and compute a if it is not specified

        self.X = X
        self.y = y

        if self.a is None and self.diameter is None:
            self.diameter = self._fast_diameter_of_point_cloud(X)

        if self.a is None:
            self.a = self.diameter / self.divisor

        return self

    def predict(self, X):
        return [self._predict_for_single_point(x) for x in X]

    def _predict_for_single_point(self, point):
        distances = [self._dist(x, point) for x in self.X]
        weights = [self._calc_weight(d) for d in distances]

        total = np.sum(weights)
        if total == 0:
            raise RuntimeError("weights total 0, a or divisor is too small, prediction failed")
        weights = [w / total for w in weights]  # now the sum of the weights is 1

        try:
            self.base_estimator.fit(self.X, self.y, weights)
        except TypeError:
            # they gave us a pipeline object, which does not allow us to pass in weights like this
            name = self.base_estimator.steps[-1][0]  # grab the last step of the pipeline, the estimator
            self.base_estimator.fit(self.X, self.y,
                                    **{name + "__sample_weight": weights})  # directly pass weights to estimator
        return self.base_estimator.predict([point])[0]

    def _dist(self, a, b):
        return euclidean_distances([a, b])[0][1]

    def _fast_diameter_of_point_cloud(self, X):
        """this is a much more complicated algorithm, but it should be much faster. :
        1. compute the centerpoint
        2. construct a cirle around this centerpoint, with a radius equal to the distance from the furthest away point from the centerpoint, and begin constricting the circle
        3. compute the distances between all points outside this constricting circle
        4. when the diameter of the constricting circle becomes less than the largest distance of the external point web, return the largest distance
        """
        centerpoint = np.mean(X, axis=0)

        distances = [self._dist(x, centerpoint) for x in X]

        vectors = list(zip(X, distances))  # okay so now every vector is a (position, distance) tuple

        vectors.sort(key=lambda x: x[1])

        furthest_distance = vectors[-1][1]

        current_highest_diameter = 0
        points_outside_hypersphere = [
            vectors.pop()[0]]  # construct a sphere around the centerpoint, with a radius sized
        # to have one point outside

        while True:
            if len(vectors) == 0:
                return current_highest_diameter

            points_intersecting_hypersphere = [vectors.pop()]

            circle_diameter = points_intersecting_hypersphere[0][1] + furthest_distance
            if current_highest_diameter > circle_diameter:
                return current_highest_diameter

            while len(vectors) > 0 and vectors[-1][1] == points_intersecting_hypersphere[0][1]:
                points_intersecting_hypersphere.append(vectors.pop())

            for point in points_intersecting_hypersphere:
                distances = [self._dist(point[0], external_point) for external_point in points_outside_hypersphere]
                max = np.max(distances)
                if max > current_highest_diameter:
                    current_highest_diameter = max
                points_outside_hypersphere.append(point[0])

            if current_highest_diameter > circle_diameter:
                return current_highest_diameter

    def _calc_weight(self, distance):
        if self.weight == "tri-cube":
            x = distance / self.a
            if x > 1: x = 1  # if we are extrapolating or if divisor > 1, we can have a point outside the hull of our
            # point cloud, producing a potentially negative or incorrectly lowered result

            return (1 - (x) ** 3) ** 3

        if self.weight == "range":
            return 1 if distance < self.a / 2 else 0

        if self.weight == "normal":
            left = 1 / np.sqrt(2 * math.pi * self.a)
            right = - (distance ** 2) / (2 * self.a)
            return left * np.exp(right)

        raise RuntimeError("self.weight " + self.weight + " is not one of tri-cube, range, normal")
