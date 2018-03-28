#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import mean_squared_error


def normalized_root_mean_square_error(true, pred):
    """
    Calculate the normalized root mean squared error.

    The error is defined as

    $$
    \frac{\sqrt{\frac{\sum _{i=1}^{n}({\hat {y}}_{i}-y_{i})^{2}}{n}}}}}
    {y_{\max} - y_{\min}}
    $$
    
    Notes
    -----
    In the case of multiple targets, the score is weighted equally between 
    the targets.

    Parameters
    ----------
    true : DataFrame
        The data frame containing the targets in the test set.
    pred : DataFrame
        The data frame containing the predicted targets.

    Returns
    -------
    error : float
        The calculated error.
    """

    errors = []

    # Loop through all the targets
    for col in pred.columns:
        intersect = true.loc[:, col].dropna().index.\
            intersection(pred.loc[:, col].dropna().index)
        
        errors.append(
            mean_squared_error(true.loc[true.index.isin(intersect), col],
                               pred.loc[pred.index.isin(intersect), col]) /
            (true.dropna().values.max() - true.dropna().values.min()))
        
    error = np.mean(errors)
    
    return error
