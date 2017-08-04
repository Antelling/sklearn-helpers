from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import SpectralClustering
import numpy as np
import warnings


class DataSimplifier(BaseEstimator, TransformerMixin):
    def __init__(self, clusterer=SpectralClustering(), percentage=None, return_old=True):
        self.clusterer = clusterer
        self.percentage = percentage
        self.return_old = return_old

    def fit_transform(self, X, Y=None, **fit_params):
        if self.percentage is not None:
            try:
                self.clusterer.set_params(n_clusters=int(len(X) * self.percentage))
            except ValueError:
                warnings.warn(
                    "Clusterer passed to DataSimplifier has not attribute n_clusters, percentage parameter has no effect")

        clusterer = self.clusterer.fit(X)
        groups = {}
        for i, label in enumerate(clusterer.labels_):
            if not label in groups:
                groups[label] = []
            groups[label].append((X[i], Y[i]) if Y is not None else X[1])

        if Y is None:
            # now we want to find the average of each group
            for group in groups:
                groups[group] = np.mean(groups[group], axis=0).tolist()

            new_x = np.array(list(groups.values()))
            if self.return_old:
                return np.concatenate([X, new_x])
            return new_x

        else:
            new_x = []
            new_y = []
            for _, group in groups.items():
                x = [g[0] for g in group]
                new_x.append(np.mean(x, axis=0).tolist())

                y = [g[1] for g in group]
                new_y.append(np.mean(y, axis=0).tolist())

            if self.return_old:
                return np.concatenate([X, new_x]), np.concatenate([Y, new_y])
            return new_x, new_y
