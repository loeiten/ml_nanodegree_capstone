#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel


class LatestDay(RegressorMixin, LinearModel):
    """
    Class for latest day prediction.

    Note
    ----
    This estimator only uses the target values for fitting and prediction.

    Attributes
    ----------
    prediction_values : array, shape (days, n_features)
        The values used for prediction.

    Examples
    --------
    >>> import numpy as np
    >>> from estimators import latest_day
    >>> reg = latest_day.LatestDay()
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y = np.array([6, 7, np.nan])
    >>> reg.fit(x, y)
    >>> reg.predict(np.array([[10, 11, 12], [13, 14, 15]]))
    array([9., 9.])
    """

    def __init__(self):
        """
        Declares member data.
        """

        # Declare member data
        self.prediction_values = None
        self._n_features = None
        self._n_targets = None

    def fit(self, x, y):
        """
        Fits the model.

        Notes
        -----
        The last value of x is being set as the prediction value
        Multi feature fitting is not yet supported. Only the last feature
        will be used in the fitting.

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
        self._n_targets = y.shape[1]

        # Initialize the self.prediction_values
        self.prediction_values = np.empty(self._n_targets)

        for col in range(self._n_targets):
            self.prediction_values[col] = x[-1, -1]

    def predict(self, x):
        """
        Perform regression on samples in x.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Set to perform regression on.

        Returns
        -------
        y_pred : array, shape (n_samples, n_features)
            Prediction values.
        """

        # Check dimension
        if x.shape[1] != self._n_features:
            raise ValueError('Dimension mismatch between fit and predict.')

        y_pred = np.repeat(self.prediction_values.copy(),
                           x.shape[0],
                           axis=0)

        return y_pred
