import numpy as np

# region stolen from https://notebooks.azure.com/coells/libraries/100days/html/day%2097%20-%20locally%20weighted%20regression.ipynb
n = 300

# generate dataset
linspace = np.linspace(-3, 3, num=n)
Y = np.log(np.abs(linspace ** 2 - 1) + .5)

# jitter X
X = linspace + np.random.normal(scale=.1, size=n)
# endregion

from lowess import LOWESS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sklearn_x = [[x] for x in X]

from matplotlib import pyplot as plt

plt.scatter(X, Y)


sklearn_linspace = [[x] for x in linspace]
for l, label in [
    (LOWESS(divisor=10), "tri-cube"),
    #(LOWESS(weight="normal", a=.1), "normal"),
    #(LOWESS(weight="range", a=1), "range"),
    (LOWESS(base_estimator=make_pipeline(PolynomialFeatures(2), LinearRegression()), divisor=10), "quartic tri-cube"),
    (LOWESS(base_estimator=make_pipeline(PolynomialFeatures(2), LinearRegression()), weight="normal", a=.1), "quartic normal"),
]:
    l.fit(sklearn_x, Y)
    predictions = l.predict(sklearn_linspace)
    plt.plot(linspace, predictions, label=label)

plt.legend()
plt.show()
