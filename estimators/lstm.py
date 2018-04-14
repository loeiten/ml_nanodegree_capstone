#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from tensorflow import set_random_seed
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
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


class LSTMRegressor(RegressorMixin, LinearModel):
    """
    Class for long short time memory.

    Attributes
    ----------
    model : keras.models.Sequential or None
        The sequential model (if created).

    Examples
    --------
    >>> import numpy as np
    >>> from estimators import lstm
    >>> reg = lstm.LSTMRegressor()
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y = np.array([6, 7, np.nan])
    >>> reg.fit(x, y)
    >>> reg.predict(np.array([[10, 11, 12], [13, 14, 15]]))
    array([[2.6165857],
           [2.9216802]], dtype=float32)
    """

    def __init__(self,
                 seed=None,
                 cells=(128,),
                 drop_out=0.0,
                 recurrent_drop_out=0.0,
                 optimizer='adam',
                 batch_size=128,
                 epochs=20,
                 time_steps=1,
                 verbose=0):
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
        verbose : int
            Verbosity level
        """

        # Set the verosity level
        self.verbose = verbose

        # Set the seed
        np.random.seed(seed)
        set_random_seed(seed)

        # Parameters to be used in make_model
        self._cells = cells
        self._drop_out = drop_out
        self._recurrent_drop_out = recurrent_drop_out
        self._optimizer = optimizer

        # Parameters to be used in self.fit
        self._epochs = epochs
        self._time_steps = time_steps
        self._batch_size = batch_size
        self._time_callback = TimeHistory()

        # Hard coded parameters (not subjected for change in this project)
        self._loss = 'mean_squared_error'

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
        x, y = prepare_input(x, y, self._time_steps)

        # Create the model
        if self.model is None:
            self.model = make_model(self._cells,
                                    x.shape,
                                    y.shape[1],
                                    self._drop_out,
                                    self._recurrent_drop_out,
                                    self._loss,
                                    self._optimizer)

        self.model.fit(x, y,
                       epochs=self._epochs,
                       batch_size=self._batch_size,
                       verbose=self.verbose,
                       callbacks=[self._time_callback],
                       shuffle=False)

    def predict(self, x):
        """
        Performs regression with the lstm model on samples in x.

        The input will be reshaped according to self.time_step.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The training data.

        Returns
        -------
        y_pred : array, shape (n_samples, n_features)
            Prediction values.
        """

        if self.model is None:
            raise RuntimeError('No model available. Build model by calling '
                               'self.fit()')

        x = prepare_input(x, None, self._time_steps)

        y_pred = self.model.predict(x)

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
    Observation with NaNs will be removed.
    Note that we lose time_step-1 targets with this method.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training data.
    y : array-like, shape (n_samples, n_targets) or None
        The target values.
    time_steps : int
        Time steps to be used in the RNN.

    Returns
    -------
    x : array-like, shape (n_samples - time_step + 1, time_step, n_features)
        The samples prepared for input to the lstm model.
    y : array-like, shape (n_samples - time_step + 1, n_targets)
        The targets prepared for input to the lstm model.
        Not returned if input y is None

    Examples
    --------
    >>> import numpy as np
    >>> nan = np.nan
    >>> x = np.array([[1, 10], [nan, 20], [3, 30], [4, 40], [5, 50], [6, 60]])
    >>> y = np.array([[2], [3], [4], [5], [6], [nan]])
    >>> time_steps = 3
    >>> prepare_input(x, y, time_steps)
    (array([[[ 1., 10.],
             [ 3., 30.],
             [ 4., 40.]],

            [[ 3., 30.],
             [ 4., 40.],
             [ 5., 50.]]]), array([[5.],
            [6.]]))
    """

    # Remove NaNs
    if y is not None:
        xy = np.hstack((x, y))
        # Slice so that observations (rows) with not NaNs are selected
        xy = xy[~np.isnan(xy).any(axis=1)]
        x = xy[:, :-y.shape[1]]
        y = xy[:, -y.shape[1]:]
    else:
        x = x[~np.isnan(x).any(axis=1)]

    batch = list()

    # Loop over the batches
    # Note that the individual batches is shifted with one day
    # +1 to include the last observation
    for b in range(x.shape[0] - time_steps + 1):
        batch.append(x[b:b+time_steps, :])

    # The size of the batch will be the first index
    x = np.asarray(batch)

    if y is not None:
        y = y[-x.shape[0]:, :]

        return x, y
    else:
        return x


def make_model(cells,
               input_shape,
               targets,
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
    input_shape : tuple, shape (samples ,time_steps, n_features)
        The shape of the input data.
        See prepare_input for details.
    targets : int
        Number of targets
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
                       input_shape=(None, input_shape[-1]),
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out,
                       return_sequences=True))
    else:
        # Create the only layer
        model.add(LSTM(cells[0],
                       input_shape=(None, input_shape[-1]),
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out))

    # Create intermediate layers
    for cell in cells[1:-1]:
        model.add(LSTM(cell,
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out,
                       return_sequences=True))

    if len(cells) > 1:
        # Create the last LSTM layer
        model.add(LSTM(cells[-1],
                       dropout=drop_out,
                       recurrent_dropout=recurrent_drop_out))

    model.add(Dense(targets))
    model.compile(loss=loss, optimizer=optimizer, metrics=[loss])

    return model
