from sklearn.datasets import load_linnerud, load_boston, load_diabetes

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from auto_curve import AutoCurver
from lowess import LOWESS
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import numpy as np

models = [
    SVR(),
    SVR(C=100, epsilon=1),
    GradientBoostingRegressor(),
    GradientBoostingRegressor(loss="lad"),
    RandomForestRegressor(),
    RandomForestRegressor(criterion="mae"),
    RandomForestRegressor(criterion="mae", max_features="sqrt"),
    KernelRidge(),
    AutoCurver(max_params=3),
    LinearRegression(),
    LOWESS(base_estimator=make_pipeline(PolynomialFeatures(2), LinearRegression()), weight="normal", divisor=50),
]

for dataset_loader in load_boston, load_diabetes:
    X, y = dataset_loader(return_X_y=True)

    scores = []
    [scores.append([]) for x in range(len(models))]

    for _ in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for i, estimator in enumerate(models):
            print(i)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            error = np.mean((predictions - y_test)**2)
            scores[i].append(error)

    print(list(np.mean(s) for s in scores))

print("\n-------------multivariate----------------\n")

models = [
    MultiOutputRegressor(SVR()),
    MultiOutputRegressor(SVR(C=100, epsilon=1)),
    MultiOutputRegressor(GradientBoostingRegressor()),
    MultiOutputRegressor(GradientBoostingRegressor(loss="lad")),
    MultiOutputRegressor(RandomForestRegressor()),
    MultiOutputRegressor(RandomForestRegressor(criterion="mae")),
    MultiOutputRegressor(RandomForestRegressor(criterion="mae", max_features="sqrt")),
    KernelRidge(),
    MultiOutputRegressor(AutoCurver(max_params=3)),
    MultiOutputRegressor(LinearRegression()),
    MultiOutputRegressor(LOWESS(weight="normal"))
]

X, y = load_linnerud(return_X_y=True)

scores = []
[scores.append([]) for x in range(len(models))]

for _ in range(30):  # try many times, since our data is so small
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    for i, estimator in enumerate(models):
        try:
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_test)
            error = np.mean(np.abs(predictions - y_test))
            scores[i].append(error)
        except Exception as e:
            print("error")
            scores.append(999999999)

print(list(np.mean(s) for s in scores))
