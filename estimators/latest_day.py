#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import RegressorMixin


class LatestDay(RegressorMixin):
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
    array([[7.],
           [7.]])
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

        Notes
        -----
        The y_test cannot contain any nans, with exception of possible nans at
         the end originating from the shift as shown in
         utils.column_modifiers.target_generator

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

        # Initialize the self.prediction_values
        self.prediction_values = np.empty(y.shape[1])

        for col in range(y.shape[1]):
            # Find the last value which is not nan
            self.prediction_values[col] = y[(~np.isnan(y)).sum(axis=0) - 1, col]

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
            raise ValueError('Dimension mismatch between fit and predict.')

        y_pred = np.repeat(self.prediction_values.copy()[np.newaxis, :],
                           x.shape[0],
                           axis=0)

        return y_pred
