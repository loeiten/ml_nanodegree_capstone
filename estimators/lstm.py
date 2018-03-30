#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.base import RegressorMixin


class LSTM(RegressorMixin):
    """
    Class for long short time memory.

    Attributes
    ----------
    FIXME
            self.optimizer = optimizer
        self.epochs = epochs
        self.time_steps = time_steps

        # Hard coded parameters (not subjected for change in this project)
        self.loss = 'mean_squared_error'

        self.model

    Examples
    --------
    FIXME
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

    def __init__(self,
                 layers=1,
                 cells=128,
                 drop_out=0.0,
                 recurrent_drop_out=0.0,
                 optimizer='adam',
                 batch_size=128,
                 epochs=20,
                 time_steps=1):
        """
        Generates a long time short memory model, sets the model.fit parameters.

        Referring to the keras documentation https://keras.io/ for more
        information about the parameters.

        Parameters
        ----------
        layers : int
            Number of hidden layers.
        cells : list
            Number of cell per layer.
            The length of the list must equal the number of layers.
        drop_out : float
            Output drop out from the output of the layer.
        recurrent_drop_out : float
            Drop out between the recurrent units.
        optimizer : str or function
            Optimizer to be used in the model.
        batch_size : int
            Size of the batch
        epochs : int
            Number of epochs
        time_steps : int
            Time steps to be used in the RNN.
        """


        # Parameters to be used in self.fit
        self.optimizer = optimizer
        self.epochs = epochs
        self.time_steps = time_steps

        # Hard coded parameters (not subjected for change in this project)
        self.loss = 'mean_squared_error'

        self.model = None

    def fit(self, x, y):
        """
        Reshapes x and y to a rank 3 tensor and fits the model.

        The input will be reshaped according to self.time_step.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The training data.
        y : array-like, shape (n_samples, n_targets)
            The target values.
        """

        y = y.copy()

        # Recast if rank 1 tensor is given
        if len(y.shape) == 1:
            y = y.reshape(len(y), 1)

        # Reshape to 3d data
        x, y = prepare_input(x, y, self.time_steps)

        self.model.fit(x, y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=2,
                       shuffle=False)

    def predict(self, x):
        """
        FIXME

        State that reshaping will happen internally

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

        # FIXME

        return y_pred


def prepare_input(x, y, time_steps):
    """
    Prepares the input for being processed by a lstm model.

    Notes
    -----
    Note that we lose time_step-1 targets with this method.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training data.
    y : array-like, shape (n_samples, n_targets)
        The target values.
    time_steps : int
        Time steps to be used in the RNN.

    Returns
    -------
    x : array-like, shape (n_samples - time_step + 1, time_step, n_features)
        The samples prepared for input to the lstm model.
    y : array-like, shape (n_samples - time_step + 1, n_targets)
        The targets prepared for input to the lstm model.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]])
    >>> y = np.array([[2], [3], [4], [5], [6], [np.nan]])
    >>> time_steps = 3
    >>> prepare_input(x, y, time_steps)
    """

    batch = list()

    # Loop over the batches
    # Note that the individual batches is shifted with one day
    # +1 to include the last observation
    for b in range(x.shape[0] - time_steps + 1):
        batch.append(x[b:b+time_steps, :])

    # The size of the batch will be the first index
    x = np.asarray(batch)

    y = y[-x.shape[0]:, :]

    return x, y
