from sklearn.datasets import load_boston as l

import numpy as np


from data_simplifier import DataSimplifier

X, y = l(return_X_y=True)

from sklearn.svm import SVR
estimator = SVR()

from sklearn import cluster as c

def test(new_X, new_y):
    estimator.fit(new_X, new_y)
    predictions = estimator.predict(X)
    error = np.mean((predictions - y) ** 2)
    return error

print("Base Score: " + str(test(X, y)))
print("")

for return_old in True, False:
    print("")
    for percentage in .05, .2, .9, .95:
        for i, clusterer in enumerate([
            c.SpectralClustering(),
            c.AgglomerativeClustering(),
            c.AffinityPropagation(),
            c.Birch(),
            c.KMeans(),
            c.MeanShift(),
        ]):
            new_X, new_y = DataSimplifier(clusterer=clusterer, percentage=percentage, return_old=return_old).fit_transform(X, y)
            score = test(new_X, new_y)
            print(str(percentage) + " " + str(i) + ": " + str(score))