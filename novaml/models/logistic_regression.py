"""
A module that contains logistic regression model.

Classes:
    LogisticRegression: A class that hold the logistic regression functions.
"""
import copy

import numpy as np

np.seterr(divide='ignore')


class LogisticRegression:
    """The logistic regression model for classification.

       Methods:
           _cost: Private method to calculate the cost function.
           _gradient_descent: Private method to calculate the gradient decent.
           train: Public method to train X and Y to get the wights and biases.
           predict: Public method to predict Y based on X and the wights and biases.
    """

    def _model(
            self,
            x: np.ndarray | float,
            w: np.ndarray | float,
            b: np.ndarray | float,
    ) -> np.ndarray | float:
        """Calculates the sigmoid function.

        Args:
            x: The x train data
            w: The weight
            b: The bias

        Returns:
            The value of f(x)
        """
        z = np.dot(x, w) + b
        fx = 1 / (1 + np.exp(-z))
        return fx

    def _cost(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray | float,
            b: np.ndarray | float,
            lambd: float | None = None,
    ) -> np.ndarray | float:
        """Calculates the cost function.

        Args:
            x: The x train data
            y: The y train data
            w: The weight
            b: The bias
            lambd: The regularization term

        Returns:
            The cost function.
        """
        m = x.shape[0]
        fx = self._model(x, w, b)
        jwb = -1 * np.sum(np.dot(y, np.log(fx)) + np.dot((1 - y), np.log(1 - fx))) / m
        if lambd:
            jwb += lambd * np.sum(np.square(w)) / (2 * m)
        return jwb

    def _gradient_descent(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray | float,
            b: np.ndarray | float,
            lambd: float | None = None,
    ) -> (np.ndarray | float, np.ndarray | float):
        """Calculate the gradient descent derivatives.

        Args:
            x: The x train data
            y: The y train data
            w: The weight
            b: The bias
            lambd: The regularization term

        Returns:
            The weight derivative and the bias derivative.
        """
        m = x.shape[0]
        fx = self._model(x, w, b)
        dw = np.sum(np.dot((fx.T - y), x)) / m
        if lambd:
            dw += lambd * w / m
        db = np.sum(fx.T - y) / m
        return dw, db

    def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w_init: np.ndarray | float,
            b_init: np.ndarray | float,
            alpha: float,
            iterations: int,
            lambd: float | None = None,
    ) -> (np.ndarray | float, np.ndarray | float, list, list):
        """Train the model to calculate the final weight and bias.

        Args:
            x: The x train data.
            y: The y train data.
            w_init: The initial weight.
            b_init: The initial bias.
            alpha: The learning rate.
            iterations: The number of iterations.
            lambd: The regularization term.

        Returns:
            The final weight, final bias, cost history, and parameters history.
        """
        cost_history = []
        parameters_history = []

        w = copy.deepcopy(w_init)
        b = copy.deepcopy(b_init)

        for i in range(iterations):
            dw, db = self._gradient_descent(x, y, w, b, lambd)
            w = w - alpha * dw
            b = b - alpha * db

            if i < 100000:
                cost_history.append(self._cost(x, y, w, b, lambd))
                parameters_history.append([w, b])

        return w, b, cost_history, parameters_history

    def predict(
            self,
            x: np.ndarray | float,
            w: np.ndarray | float,
            b: np.ndarray | float,
    ) -> np.ndarray | float:
        """Predict the new Yhat given the X and final weight and bias.

        Args:
            x: The x train data.
            w: The final weight.
            b: The final bias.

        Returns:
            The predicted yhat.
        """
        yhat = self._model(x, w, b)
        return yhat
