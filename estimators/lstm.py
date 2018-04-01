#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn.base import RegressorMixin
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import Callback


class TimeHistory(Callback):
    """
    A custom callback for timing the epochs

    Source
    https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit

    Attributes
    ----------
    times : list
        The elapsed time for each training epoch.
    """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class LSTMRegressor(RegressorMixin):
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
                 cells=(128,),
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
        cells : list or tuple
            Number of cell per LSTM layer.
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

        # Parameters to be used in make_model
        self.cells = cells
        self.drop_out = drop_out
        self.recurrent_drop_out = recurrent_drop_out
        self.optimizer = optimizer

        # Parameters to be used in self.fit
        self.optimizer = optimizer
        self.epochs = epochs
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.time_callback = TimeHistory()

        # Hard coded parameters (not subjected for change in this project)
        self.loss = 'mean_squared_error'
        self.stateful = True

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

        # Create the model
        if self.model is None:
            self.model = make_model(self.cells,
                                    x.shape,
                                    self.stateful,
                                    self.drop_out,
                                    self.recurrent_drop_out,
                                    self.loss,
                                    self.optimizer)

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
        y_pred = None
        return y_pred

    def reset_model(self):
        """
        Resets the model
        """

        self.model = None


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
    (array([[[ 1, 10],
             [ 2, 20],
             [ 3, 30]],

            [[ 2, 20],
             [ 3, 30],
             [ 4, 40]],

            [[ 3, 30],
             [ 4, 40],
             [ 5, 50]],

            [[ 4, 40],
             [ 5, 50],
             [ 6, 60]]]),
     array([[ 4.],
            [ 5.],
            [ 6.],
            [nan]]))
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


def make_model(cells,
               shape,
               stateful,
               drop_out,
               recurrent_drop_out,
               loss,
               optimizer):
    """
    Returns a long short time memory model.

    Parameters
    ----------
    cells : list or tuple
        The number of cells (similar to neurons) for each lstm layer.
    shape : tuple, shape (samples ,time_steps, n_features)
        The shape of the input data.
        See prepare_input for details.
    stateful : bool
        Whether the lstm should be stateful or not
    drop_out : float
        Output drop out from the output of the layer.
    recurrent_drop_out : float
        Drop out between the recurrent units.
    loss : str
        The loss function to be used.
    optimizer : str or function
        Optimizer to be used in the model.

    Returns
    -------
    model : keras.models.Sequential
        The sequential model.
    """

    model = Sequential()

    # Create the initial LSTM layer
    if len(cells) > 1:
        model.add(LSTM(cells[0],
                       batch_input_shape=shape,
                       stateful=stateful,
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out,
                       return_sequences=True))
    else:
        # Create the only layer
        model.add(LSTM(cells[0],
                       batch_input_shape=shape,
                       stateful=stateful,
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out))

    # Create intermediate layers
    for cell in cells[1:-1]:
        model.add(LSTM(cell,
                       stateful=stateful,
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out,
                       return_sequences=True))

    if len(cells) > 1:
        # Create the last LSTM layer
        model.add(LSTM(cells[-1],
                       stateful=stateful,
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out))

    # Output: 7 day, 14 day and 28 day prediction
    model.add(Dense(3))
    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])

    return model
