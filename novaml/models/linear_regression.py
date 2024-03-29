"""
A module that contains linear regression model.

Classes:
    LinearRegression: A class that hold the linear regression functions.
"""
import copy

import numpy as np


class LinearRegression:
    """The linear regression model for training and predicting linear models.

       Methods:
           _model: Private method to calculate the model.
           _cost: Private method to calculate the cost function.
           _gradient_descent: Private method to calculate the gradient decent.
           train: Public method to train X and Y to get the wights and biases.
           predict: Public method to predict Y based on X and the wights and biases.
    """

    def _model(
            self,
            x: np.ndarray,
            w: np.ndarray | float,
            b: np.ndarray | float,
    ) -> np.ndarray | float:
        """Calculates the linear model.

        Args:
            x: The x train data.
            w: The weight.
            b: The bias.

        Returns:
            The value of f(x).
        """
        fx = np.dot(x, w) + b
        return fx

    def _cost(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray | float,
            b: np.ndarray | float,
            lambd=None,
    ) -> np.ndarray | float:
        """Calculate the cost function.

        Args:
            x: The x train data.
            y: The y train data.
            w: The weight.
            b: The bias.
            lambd: The regularization term.

        Returns:
            The cost function.
        """
        m = x.shape[0]
        fx = self._model(x, w, b)
        jwb = np.sum(np.square(fx - y)) / (2 * m)
        if lambd:
            jwb += lambd * np.sum(w ** 2) / (2 * m)
        return jwb

    def _gradient_descent(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray | float,
            b: np.ndarray | float,
            lambd=None,
    ) -> (np.ndarray | float, np.ndarray | float):
        """Calculate the gradient descent derivatives.

        Args:
            x: The x train data.
            y: The y train data.
            w: The weight.
            b: The bias.
            lambd: The regularization term.

        Returns:
            The weight derivative and the bias derivative.
        """
        m = x.shape[0]
        fx = self._model(x, w, b)
        dw = np.sum(np.dot((fx - y), x)) / m
        if lambd:
            dw += lambd * w / m
        db = np.sum(fx - y) / m
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
