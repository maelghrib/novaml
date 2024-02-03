import numpy as np

from novaml.models import LogisticRegression

x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

logistic_regression = LogisticRegression()

final_w, final_b, _, _ = logistic_regression.train(
    x=x_train,
    y=y_train,
    w_init=0,
    b_init=0,
    alpha=0.01,
    iterations=1000,
    lambd=0,
)

yhat = logistic_regression.predict(x=x_train, w=final_w, b=final_b)

print(f"yhat: {yhat}")
print(f"ytrain: {y_train}")
