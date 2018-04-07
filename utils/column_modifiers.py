#!/usr/bin/env python
# -*- coding: utf-8 -*-


def target_generator(df, shift_column, target_days, copy=True):
    """
    Generates the targets

    Parameters
    ----------
    df : DataFrame
        The data frame to make the targets on
    shift_column : object
        The column to generate the targets from
    target_days : list
        List of days to shift
    copy : bool
        If a copy is returned

    Returns
    -------
    df : DataFrame
        The data frame with the targets

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from utils.column_modifiers import target_generator
    >>> df = pd.DataFrame(np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]),
    ...                   columns=['a', 'b', 'c'])
    >>> target_generator(df, 'c', [1, 2])
        a   b   c  c + 1 days  c + 2 days
    0   1   2   3         6.0         9.0
    1   4   5   6         9.0        12.0
    2   7   8   9        12.0         NaN
    3  10  11  12         NaN         NaN
    """
    if copy:
        df = df.copy()

    for days in target_days:
        df[f'{shift_column} + {days} days'] = \
            df.loc[:, shift_column].shift(-days)

    return df


def feature_generator(df, shift_column, days, copy=True):
    """
    Generates features

    Attempts to insert the columns immediately before the position of
    shift_column

    Parameters
    ----------
    df : DataFrame
        The data frame to make the targets on
    shift_column : str
        The column to generate the targets from
    days : int
        Number of days to create features from
    copy : bool
        If a copy is returned

    Returns
    -------
    df : DataFrame
        The data frame with the targets

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from utils.column_modifiers import feature_generator
    >>> df = pd.DataFrame(np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]),
    ...                   columns=['a', 'b', 'c'])
    >>> feature_generator(df, 'b', 3)
          a  b - 3 days  b - 2 days  b - 1 days     b     c
    0   1.0         NaN         NaN         NaN   2.0   3.0
    1   4.0         NaN         NaN         2.0   5.0   6.0
    2   7.0         NaN         2.0         5.0   8.0   9.0
    3  10.0         2.0         5.0         8.0  11.0  12.0
    """
    if copy:
        df = df.copy()

    for day in range(1, days+1):
        # Get last column
        cols = df.columns
        last_ind = [el for el in cols if str(shift_column) in str(el)][0]
        ind = list(cols).index(last_ind)
        col = df.loc[:, shift_column].shift(day)
        df.insert(ind, f'{shift_column} - {day} days', col)

    return df


def keep_columns(df, columns, copy=True):
    """
    Removes all columns except those specified in columns

    Parameters
    ----------
    df : DataFrame
        The data frame to make the targets on
    columns : list
        List of columns to keep
    copy : bool
        If a copy is returned

    Returns
    -------
    df : DataFrame
        The data frame with the targets

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from utils.column_modifiers import keep_columns
    >>> df = pd.DataFrame(np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]),
    ...                   columns=['a', 'b', 'c'])
    >>> keep_columns(df, ['a', 'c'])
          a     c
    0   1.0   3.0
    1   4.0   6.0
    2   7.0   9.0
    3  10.0  12.0
    """
    if copy:
        df = df.copy()

    remove = [col for col in df.columns if col not in columns]

    df.drop(remove, axis=1, inplace=True)

    return df
