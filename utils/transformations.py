#!/usr/bin/env python
# -*- coding: utf-8 -*-


class StockMinMax(object):
    """
    MinMax scaler for stocks.

    Attributes
    ----------
    max : pandas.core.series.Series
        The maximum value of the clean features (i.e. without - or + in the
        names), neglecting nans
    min : pandas.core.series.Series
        The minimum value of the clean features (i.e. without - or + in the
        names), neglecting nans

    Examples
    --------
    FIXME

    """

    def __init__(self):
        """
        Declares the member variables.
        """

        self.max = None
        self.min = None

    def fit(self, x):
        """
        Fit the variables of the scaler.

        Only 'clean' features will be taken into account.
        I.e. if x contains the features 'foo', 'foo - 1', 'bar', only 'foo'
        and 'bar' will be fitted.

        Parameters
        ----------
        x : DataFrame
            The fitting data
        """

        clean_cols = [col for col in x.columns
                      if '-' not in col and '+' not in col]

        self.max = x.loc[:, clean_cols].max(skipna=True)
        self.min = x.loc[:, clean_cols].min(skipna=True)

    def transform(self, x):
        """
        Transforms x.

        Features will be scaled according to the features in self.max and
        self.min by

        x' = (x - x_min)/(x_max - x_min)

        Parameters
        ----------
        x : DataFrame
            The data frame to transform.

        Returns
        -------
        x_prime : DataFrame
            The transformed data.
        """

        x_prime = x.copy()

        x_cols = x.columns
        for max_col in self.max.index:
            for x_col in x_cols:
                if max_col == x_col[:len(max_col)]:
                    x_prime.loc[:, x_col] =\
                        (x.loc[:, x_col] - self.min.loc[max_col]) / \
                        (self.max.loc[max_col] - self.min.loc[max_col])

        return x_prime

    def inverse_transform(self, x_prime):
        """
        Inverse transforms x.

        Features will be scaled according to the features in self.max and
        self.min by

         x = x'*(x_max - x_min) + x_min

        Parameters
        ----------
        x_prime : DataFrame
            The data frame to transform.

        Returns
        -------
        x : DataFrame
            The transformed data.
        """

        x = x_prime.copy()

        x_cols = x_prime.columns
        for max_col in self.max.index:
            for x_col in x_cols:
                if max_col == x_col[:len(max_col)]:
                    x.loc[:, x_col] = \
                        x_prime.loc[:, x_col] * \
                        (self.max.loc[max_col] - self.min.loc[max_col])\
                        + self.min.loc[max_col]

        return x


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