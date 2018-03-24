#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


class RandomGaussian(object):
    """
    Class for random Gaussian prediction.

    Attributes
    ----------
    days : int
        Number of days to do the prediction on.
    mean : array-like, shape (n_features,)
        The mean of the fitted data.
    mean : array-like, shape (n_features,)
        The standard deviation of the fitted data.

    Notes
    -----
    This estimator do not use target values.

    Examples
    --------
    >>> import numpy as np
    >>> from estimators import random_gaussian
    >>> reg = random_gaussian.RandomGaussian(days=2, seed=42)
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> reg.fit(x)
    >>> reg.predict(np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]]))
    array([[3.24507123, 3.29260355, 5.47153281],
           [4.78454478, 3.14876994, 4.14879456]])
    """

    def __init__(self, days=7, seed=None):
        """
        Initialized the estimator with number of days to predict.

        Also declares member data.

        Parameters
        ----------
        days : int
            Number of days to do the prediction on.
        seed : None or int
            Seed for the random number generator
        """

        # Set the seed
        np.random.seed(seed)

        self.days = days

        # Declare member data
        self.mean = None
        self.std = None
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
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

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

        y_pred = np.random.normal(self.mean,
                                  self.std,
                                  (self.days, len(self.mean)))

        return y_pred
