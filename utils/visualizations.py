#!/usr/bin/env python
# -*- coding: utf-8 -*-


from utils.column_modifiers import reshift_targets


def plot_true_and_prediction(true, pred, columns=None, y_label=''):
    """
    Plots the true values against the predictions

    Parameters
    ----------
    true : DataFrame
        The data frame containing the true values.
        Note that in our case this is x_test.
    pred : DataFrame
        The data frame containing the predicted targets.
    columns : None or list
        If None, all columns will be plotted.
        If list: A list of the column numbers to be plotted.
    y_label : str
        The y_label of the plot

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes
    """

    ax = true.plot()

    df = reshift_targets(pred, true.index)

    if columns is None:
        _ = df.plot(ax=ax)
    else:
        for col in columns:
            _ = df.loc[:, [col]].plot(ax=ax)

    ax.grid()
    _ = ax.set_ylabel(y_label)
    
    return ax
