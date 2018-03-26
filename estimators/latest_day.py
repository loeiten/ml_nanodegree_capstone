#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import RegressorMixin


class LatestDay(RegressorMixin):
    """
    Class for latest day prediction.

    Attributes
    ----------
    prediction_values : array, shape (days, n_features)
        The values used for prediction.

    Notes
    -----
    This estimator only uses the target values for fitting and prediction.

    Examples
    --------
    >>> import numpy as np
    >>> from estimators import latest_day
    >>> reg = latest_day.LatestDay()
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([7, 8])
    >>> reg.fit(x, y)
    >>> reg.predict(np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]]))
    array([8, 8, 8])
    """

    def __init__(self):
        """
        Declares member data.
        """

        # Declare member data
        self.prediction_values = None
        self._n_features = None

    def fit(self, x, y):
        """
        Fits the model.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The training data.
        y : array-like, shape (n_samples, n_targets)
            The target values.
        """

        self._n_features = x.shape[1]

        y = y.copy()

        # Recast if rank 1 tensor is given
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        self.prediction_values = y[-1, :]

    def predict(self, x):
        """
        Perform classification on samples in x.

        Parameters
        ----------
        x : array-like, shape (_, n_features)
            Set to perform regression on.

        Returns
        -------
        y_pred : array, shape (_, n_features)
            Prediction values.
        """

        # Check dimension
        if x.shape[1] != self._n_features:
            raise ValueError("Dimension mismatch between fit and predict.")

        y_pred = self.prediction_values.copy()[np.newaxis, :]

        return y_pred
