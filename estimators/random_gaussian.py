#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import RegressorMixin


class RandomGaussian(RegressorMixin):
    """
    Class for random Gaussian prediction.

    Note
    ----
    This estimator only uses the target values for fitting and prediction.

    Attributes
    ----------
    mean : array-like, shape (n_features,)
        The mean of the fitted data.
    std : array-like, shape (n_features,)
        The standard deviation of the fitted data.

    Examples
    --------
    >>> import numpy as np
    >>> from estimators import random_gaussian
    >>> reg = random_gaussian.RandomGaussian(seed=42)
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y = np.array([6, 7, np.nan])
    >>> reg.fit(x, y)
    >>> reg.predict(np.array([[10, 11, 12], [13, 14, 15]]))
    array([[6.74835708],
           [6.43086785]])
    """

    def __init__(self, seed=None):
        """
        Initialized the estimator with number of days to predict.

        Also declares member data.

        Parameters
        ----------
        seed : None or int
            Seed for the random number generator
        """

        # Set the seed
        np.random.seed(seed)

        # Declare member data
        self.mean = None
        self.std = None
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

        self.mean = np.nanmean(y, axis=0)
        self.std = np.nanstd(y, axis=0)

    def predict(self, x):
        """
        Perform regression on samples in x.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Set to perform classification on.

        Returns
        -------
        y_pred : array, shape (n_samples, n_features)
            Prediction values.
        """

        # Check dimension
        if x.shape[1] != self._n_features:
            raise ValueError('Dimension mismatch between fit and predict.')

        y_pred = np.random.normal(self.mean,
                                  self.std,
                                  (x.shape[0], len(self.mean)))

        return y_pred
