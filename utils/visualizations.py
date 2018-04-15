#!/usr/bin/env python
# -*- coding: utf-8 -*-


from utils.column_modifiers import reshift_targets
import matplotlib.pyplot as plt


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

    ax = None

    if columns is None:
        ax = true.plot(alpha=0.7)
    else:
        for col in columns:
            if ax is not None:
                _ = true.loc[:, [col]].plot(ax=ax)
            else:
                ax = true.loc[:, [col]].plot(alpha=0.7)

    df = reshift_targets(pred, true.index)
    df_cols = df.columns

    if columns is None:
        _ = df.plot(ax=ax, alpha=0.7)
    else:
        for df_col in df_cols:
            for col in columns:
                if col in df_col:
                    _ = df.loc[:, [df_col]].plot(ax=ax, alpha=0.7)

    ax.grid()
    _ = ax.set_ylabel(y_label)
    
    return ax


def plot_scores(training_socres, validation_scores, title=''):
    """
    Plots the scores

    Parameters
    ----------
    training_scores : dict
        Dictionary on the form
        >>> {stock : {day : score}}
        Where stock is a string, day is a int, and score is a float
    validation_scores : dict
        Dictionary on the form
        >>> {stock : {day : score}}
        Where stock is a string, day is a int, and score is a float
    title : str
        Title for the plot

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes
    """

    _, ax = plt.subplots()
    colors = list()

    for key in training_socres.keys():
        lines = ax.plot(validation_scores[key].keys(),
                        validation_scores[key].values(),
                        alpha=0.7,
                        label=key + ' validation')
        colors.append(lines[0].get_color())

    for key, color in zip(training_socres.keys(), colors):
        _ = ax.plot(training_socres[key].keys(),
                    training_socres[key].values(),
                    linestyle='--',
                    color=color,
                    alpha=0.7,
                    label=key + ' training')

    ax.grid()
    _ = ax.set_xlabel('Days')
    _ = ax.set_ylabel('Error')
    ax.legend(loc='best', fancybox=True, framealpha=0.5, ncol=2)
    ax.set_title(title)

    return ax
