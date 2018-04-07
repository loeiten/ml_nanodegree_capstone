#!/usr/bin/env python
# -*- coding: utf-8 -*-


def subtract_previous_row(df, cols=None, copy=False):
    """
    Subtract the previous row with the current row.

    Note that the transformation of a row depends on the previous row,
    meaning that pandas.transform or pandas.apply are not suitable for this
    operation.

    Notes
    -----
    The first row will have a NaN value as the -1st row does not exist.
    As the transformation here is t_i = x_i - x_{i-1}, the back
    transformation is dependent on the true previous prediction as
    x_i = x_{i-1} + t_i. I.e. the transformation only makes sense for
    prediction of the immediate next row.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to modify
    cols : None or list
        If None, the transformation will be applied to all columns.
        If list the column in the list will be transformed.

    Returns
    -------
    df : DataFrame
        The modified DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from utils.transformations import subtract_previous_row
    >>> df = pd.DataFrame(np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]),
    ...                   columns=['a', 'b', 'c'])
    >>> subtract_previous_row(df, ['a', 'c'])
         a   b    c
    0  NaN   2  NaN
    1  3.0   5  3.0
    2  3.0   8  3.0
    3  3.0  11  3.0
    """

    if copy:
        df = df.copy()

    if cols is None:
        cols = df.columns

    for col in cols:
        col_vals = df.loc[:, col]
        prev_col_vals = col_vals.shift(1)
        df.loc[:, col] = col_vals - prev_col_vals

    return df
