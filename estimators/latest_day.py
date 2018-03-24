#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


class LatestDay(object):
    """
    Class for latest day prediction.

    Attributes
    ----------
    days : int
        Number of days to do the prediction on.
    prediction_values : array, shape (days, n_features)
        The values used for prediction.

    Notes
    -----
    This estimator do not use target values.

    Examples
    --------
    >>> import numpy as np
    >>> from estimators import latest_day
    >>> reg = latest_day.LatestDay(days=2)
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> reg.fit(x)
    >>> reg.predict(np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]]))
    array([[4, 5, 6],
           [4, 5, 6]])
    """

    def __init__(self, days=7):
        """
        Initialized the estimator with number of days to predict.

        Also declares member data.

        Parameters
        ----------
        days : int
            Number of days to do the prediction on.
        """

        self.days = days

        # Declare member data
        self.prediction_values = None
        self._n_features = None

    def fit(self, x):
        """
        Fits the model.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Set of samples, where n_samples is the number of samples and
            n_features is the number of features.
        """

        self._n_features = x.shape[1]
        self.prediction_values = x[-1, :].copy()

    def predict(self, x):
        """
        Perform classification on samples in x.

        Parameters
        ----------
        x : array-like, shape (_, n_features)
            Set to perform classification on.

        Returns
        -------
        y_pred : array, shape (days, n_features)
            Prediction values.
        """

        # Check dimension
        if x.shape[1] != self._n_features:
            raise ValueError("Dimension mismatch between fit and predict.")

        y_pred = np.repeat(self.prediction_values.copy()[np.newaxis, :],
                           self.days,
                           axis=0)

        return y_pred
