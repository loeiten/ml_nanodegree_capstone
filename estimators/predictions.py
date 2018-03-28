#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np


def calculate_rolling_prediction(reg, x_train, x_test, y_train, y_test):
    """
    Returns a rolling prediction.

    Our goal is to make a prediction every time new data becomes available to
    us. We will retrain the model before each prediction.

    The first prediction will use only the data from the training set, and the
    prediction will be for a date (or dates) belonging to the test set. However,
    this means that we have a lot of unused data in the test set, so we can
    do a rolling prediction.

    I.e. we will update the training set with the earliest observation from the
    test set and make the next prediction. Then we will update the training set
    with the two earliest observations in the test set and so on, until we
    have predicted until the latest date in the test set.

    Notes
    -----
    The y_test cannot contain any nans, with exception of possible nans at the
    end originating from the shift as shown in
    utils.column_modifiers.target_generator

    Parameters
    ----------
    reg : object
        The model to fit and make prediction from
    x_train : DataFrame, shape (n_train_samples, n_features)
        The features of the training set
    x_test :  DataFrame, shape (n_test_samples, n_features)
        The features of the test set
    y_train : DataFrame, shape (n_train_samples, n_targets)
        The targets of the training set
    y_test :  DataFrame, shape (n_test_samples, n_targets)
        The targets of the test set

    Returns
    -------
    pred_df : DataFrame
        The DataFrame containing the predictions.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import linear_model
    >>> from utils.column_modifiers import target_generator
    >>> df = pd.DataFrame(np.array(range(16)), columns=['x'])
    >>> df = target_generator(df, 'x', [2, 3])
    >>> df.tail()
         x  x + 2 days  x + 3 days
    11  11        13.0        14.0
    12  12        14.0        15.0
    13  13        15.0         NaN
    14  14         NaN         NaN
    15  15         NaN         NaN
    >>> x = df.loc[:, ['x']]
    >>> y = df.loc[:, ['x + 2 days', 'x + 3 days']]
    >>> x_train, x_test, y_train, y_test = \
    ...     train_test_split(x, y, shuffle=False, test_size=12)
    >>> reg = linear_model.LinearRegression()
    >>> calculate_rolling_prediction(reg, x_train, x_test, y_train, y_test)
        x + 2 days  x + 3 days
    5          5.0         NaN
    6          6.0         6.0
    7          7.0         7.0
    8          8.0         8.0
    9          9.0         9.0
    10        10.0        10.0
    11        11.0        11.0
    12        12.0        12.0
    13        13.0         NaN
    """

    # Initialize the DataFrames list
    df_list = []

    # Obtain the day of prediction
    prediction_days = y_test.isnull().sum()

    # Do the rolling prediction for each of the columns in y
    for col_ind in range(len(y_test.columns)):
        y_train_cur_col = y_train.loc[:, y_train.columns[col_ind]]
        y_test_cur_col = y_test.loc[:, y_test.columns[col_ind]]
        days = prediction_days[col_ind]

        # Initialize y_pred
        # The length of the array will be the same as y_test, but -days
        # shorter due to the prediction is done "days" days in the future,
        # and additional -days shorter accounting for the "days" NaN values
        # at the end of y_test (which arises due to the shift)
        # The +1 comes from the first prediction which uses only the training
        # set
        y_pred = np.empty(y_test_cur_col.shape[0] - 2*days + 1)

        for pred_nr in range(len(y_pred)):
            # Extend the training data
            rolling_x = pd.concat([x_train, x_test.iloc[:pred_nr]], axis=0)
            rolling_y = pd.concat([y_train_cur_col,
                                   y_test_cur_col.iloc[:pred_nr]], axis=0)
            # Fit (retrain) the model with the rolling_features set
            reg.fit(rolling_x.values, rolling_y.values)
            # Make prediction and append to y_pred
            # We are after the latest value in the prediction as this
            # corresponds to the prediction of the latest date in rolling_x
            y_pred[pred_nr] = reg.predict(rolling_x.values)[-1]

        # Cast the result into a DataFrame for easier post-processing
        # The indexing from y_test includes the first days where there will
        # be no prediction for (except the one done from the pure trainng
        # set), and the nan values at the end of y_test due to the shift
        df_list.append(pd.DataFrame(y_pred,
                       index=y_test_cur_col.index[days - 1: -days],
                       columns=[y_test_cur_col.name + ' predicted']))

    pred_df = pd.concat(df_list, axis=1)

    return pred_df