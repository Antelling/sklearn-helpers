from sklearn.metrics.pairwise import euclidean_distances


def _dist(a, b):
    return euclidean_distances([a, b])[0][1]


def _fast_diameter_of_point_cloud(X):
    """this is a much more complicated algorithm, but it should be much faster. :
    1. compute the centerpoint
    2. construct a cirle around this centerpoint, with a radius equal to the distance from the furthest away point from the centerpoint, and begin constricting the circle
    3. compute the distances between all points outside this constricting circle
    4. when the diameter of the constricting circle becomes less than the largest distance of the external point web, return the largest distance
    """
    centerpoint = np.mean(X, axis=0)

    distances = [_dist(x, centerpoint) for x in X]

    vectors = list(zip(X, distances))  # okay so now every vector is a (position, distance) tuple

    vectors.sort(key=lambda x: x[1])

    furthest_distance = vectors[-1][1]

    current_highest_diameter = 0
    points_outside_hypersphere = [vectors.pop()[0]]  # construct a sphere around the centerpoint, with a radius sized
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
            distances = [_dist(point[0], external_point) for external_point in points_outside_hypersphere]
            max = np.max(distances)
            if max > current_highest_diameter:
                current_highest_diameter = max
            points_outside_hypersphere.append(point[0])

        if current_highest_diameter > circle_diameter:
            return current_highest_diameter





import numpy as np
from time import time

samples = np.linspace(10, 1000000, 10)

times = []

for n_samples in samples:
    start_time = time()
    cloud = [np.random.randint(0, 100, 5) for _ in range(int(n_samples))]
    endtime = time()
    times.append(endtime - start_time)

from matplotlib import pyplot as plt
plt.plot(samples, times)
plt.show()
