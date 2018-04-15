#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor


def calculate_rolling_prediction(reg,
                                 x_train,
                                 x_test,
                                 y_train,
                                 y_test,
                                 prediction_days,
                                 training_prediction=False):
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
    The training prediction in the rolling prediction predicts on the last
    rolled value, whereas the training prediction in the normal prediction
    predicts on the input training set.

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
    prediction_days : array-like, shape (n_targets)
        The number of days the targets aim to predict
    training_prediction : bool
        Whether or not predictions should be made on the training set

    Returns
    -------
    pred_df : DataFrame, shape (n_test_samples - min_nans, n_targets)
        The DataFrame containing the predictions.
        The minimum amount of trailing NaNs will be stripped from the end.
    train_pred_df : DataFrame, shape (n_test_samples - min_nans, n_targets)
        The DataFrame containing the predictions on the fitted training set.
        Is shifted one index compared to the pred_df.
        The minimum amount of trailing NaNs will be stripped from the end.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import linear_model
    >>> from utils.column_modifiers import target_generator
    >>> from estimators.predictions import calculate_rolling_prediction
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
    pred_list = list()

    if training_prediction:
        train_pred_list = list()

    # If feature_generator has been used, then NaNs have been
    # introduced in the training set. We remove these
    introduced_nans = x_train.isnull().sum().max()
    x_train = x_train.iloc[introduced_nans:]
    y_train = y_train.iloc[introduced_nans:]

    # Do the rolling prediction for each of the columns in y
    for col_ind in range(len(y_test.columns)):
        # Obtain the columns
        y_train_cur_col = y_train.loc[:, y_train.columns[col_ind]]
        y_test_cur_col = y_test.loc[:, y_test.columns[col_ind]]

        # How many days in the future we are predicting for
        days = prediction_days[col_ind]

        # Initialize y_pred
        # The length of the array will be the same as y_test, but -days
        # shorter due to the NaN values at the end of y_test (which arises
        # due to the shift)
        y_pred = np.empty(y_test_cur_col.shape[0] - days)

        if training_prediction:
            y_train_pred = np.empty(y_test_cur_col.shape[0] - days)

        for pred_nr in range(len(y_pred)):
            # Extend the training data
            # NOTE: x_test.iloc[:0] will return an empty DataFrame
            rolling_x = pd.concat([x_train, x_test.iloc[:pred_nr]],
                                  axis=0)
            rolling_y = pd.concat([y_train_cur_col,
                                   y_test_cur_col.iloc[:pred_nr]],
                                  axis=0)

            # Fit (retrain) the model with the rolling_features set
            reg.fit(rolling_x.values, rolling_y.values)
            # Make prediction and append to y_pred
            # Predict for test value after the test value which was just
            # included to rolling_x
            # NOTE: There are as many predictions as the length of y_pred.
            #       The first prediction is using data only from the training
            #       set.
            #       x_test.iloc[0] returns the first element
            y_pred[pred_nr] = \
                reg.predict(x_test.iloc[pred_nr].values[np.newaxis, :])[-1]

            if training_prediction:
                # NOTE: When fitting, we use the data until (but no
                #       including) the pred_nr index of x_test.
                #       y_pred_train is making a prediction on this element,
                #       whereas y_pred is making a prediction on the next
                #       element
                y_train_pred[pred_nr] = \
                    reg.predict(rolling_x.iloc[-1].values[np.newaxis, :])[-1]

        # Cast the result into a DataFrame for easier post-processing
        # The indexing from y_test includes the first days where there will
        # be no prediction for (except the one done from the pure training
        # set), and the nan values at the end of y_test due to the shift
        pred_list.append(pd.DataFrame(y_pred,
                         index=y_test_cur_col.index[: -days],
                         columns=[y_test_cur_col.name + ' predicted']))

        if training_prediction:
            index = list([y_train_cur_col.index[-1],
                          *y_test_cur_col.index[: -days - 1]])
            train_pred_list.append(
                pd.DataFrame(y_train_pred,
                             index=index,
                             columns=[y_test_cur_col.name + ' fit prediction']))

    pred_df = pd.concat(pred_list, axis=1)
    if training_prediction:
        train_pred_df = pd.concat(train_pred_list, axis=1)

        return pred_df, train_pred_df

    return pred_df


def calculate_normal_prediction(reg,
                                x_train,
                                x_test,
                                y_train,
                                y_test,
                                prediction_days,
                                training_prediction=False,
                                use_multi_output_regressor=True,
                                consistent_with_rolling=True):
    """
    Returns a normal prediction.

    In this prediction, we will not take into account new information as it
    becomes available to us, but make a prediction for the whole test set.

    Notes
    -----
    The training prediction in the rolling prediction predicts on the last
    rolled value, whereas the training prediction in the normal prediction
    predicts on the input training set.
    Hence, consistent_with_rolling will not have any effect on train_pred_df.

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
    prediction_days : array-like, shape (n_targets)
        The number of days the targets aim to predict.
        Is only effective when consistent_with_rolling is True
    training_prediction : bool
        Whether or not predictions should be made on the training set
    use_multi_output_regressor : bool
        Use sklearn's multi output regressor. This must be used when the
        estimator does not support multiple predictions out of the box.
    consistent_with_rolling : bool
        The output will be on the same form as that obtained from
        calculate_rolling_prediction.
        Will not have any effect on train_pred_df (see notes above).

    Returns
    -------
    pred_df : DataFrame
        The DataFrame containing the predictions.
    train_pred_df : DataFrame, shape (n_test_samples - min_nans, n_targets)
        The DataFrame containing the predictions on the fitted training set.
        Is shifted one index compared to the pred_df.
        The minimum amount of trailing NaNs will be stripped from the end.
        consistent_with_rolling will not have any effect on this output (see
        notes above).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import linear_model
    >>> from utils.column_modifiers import target_generator
    >>> from estimators.predictions import calculate_normal_prediction
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
    >>> calculate_normal_prediction(reg, x_train, x_test, y_train, y_test)
        x + 2 days predicted  x + 3 days predicted
    4                    6.0                   7.0
    5                    7.0                   8.0
    6                    8.0                   9.0
    7                    9.0                  10.0
    8                   10.0                  11.0
    9                   11.0                  12.0
    10                  12.0                  13.0
    11                  13.0                  14.0
    12                  14.0                  15.0
    13                  15.0                   NaN
    """

    if use_multi_output_regressor:
        reg = MultiOutputRegressor(reg)

    # If feature_generator has been used, then NaNs have been
    # introduced in the training set. We remove these
    introduced_nans = x_train.isnull().sum().max()
    x_train = x_train.iloc[introduced_nans:]
    y_train = y_train.iloc[introduced_nans:]

    reg.fit(x_train.values, y_train.values)

    y_pred = reg.predict(x_test.values)

    if training_prediction:
        y_train_pred = reg.predict(x_train.values)

    # Cast the result into a DataFrame for easier post-processing
    # The indexing from y_test includes the first days where there will
    # be no prediction for (except the one done from the pure training
    # set), and the nan values at the end of y_test due to the shift
    columns = [col + ' predicted' for col in y_test.columns]

    pred_df = pd.DataFrame(y_pred,
                           index=y_test.index,
                           columns=columns)

    if training_prediction:
        train_pred_df = pd.DataFrame(y_train_pred,
                                     index=y_train.index,
                                     columns=columns)

    if consistent_with_rolling:
        # Replace with NaNs (we are actually making more predictions than we
        # need)
        for days, col in zip(prediction_days, columns):
            pred_df.loc[pred_df.index[-days:], col] = np.nan
            if training_prediction:
                train_pred_df.loc[train_pred_df.index[-days:], col] = np.nan

        # Remove predictions where we do not have any targets
        pred_df = pred_df.iloc[:-prediction_days.min()]

        if training_prediction:
            train_pred_df = train_pred_df.iloc[:-prediction_days.min()]

    if training_prediction:
        # Rename columns for training prediction
        columns = [col + ' fit prediction' for col in y_test.columns]

        train_pred_df.columns = columns

        return pred_df, train_pred_df

    return pred_df
