#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


class OCHLVAData(object):
    """
    Class treating Open, Close, High, Low, Volume and Adjusted close

    Attributes
    ----------
    raw_data : dict
        Dictionary of the data frames of the raw stock data.
    clean_data : dict
        Dictionary of the data frames of the cleaned stock data.
    transformed_data : dict
        Dictionary of the transformed data frames.
    dates : Index
        The dates S&P 500 was traded on

    Examples
    --------
    The following example will load AAPL and plot its Close and Adj. Close
    together with S&P 500

    >>> data = OCHLVAData()  # Loads S&P500 as ^GSCP for reference dates
    >>> data.load_data('AAPL')
    >>> data.plot(['Close', 'Adj. Close']
    """

    def __init__(self):
        """
        Constructor which loads the ^GSPC ticker in order to get the valid
        time range.
        """

        self.data_dir = Path(__file__).absolute().parents[1].joinpath("data")
        symbols = [e.name.replace('.csv', '')
                   for e in self.data_dir.glob('*.csv')]
        self.available_symbols = tuple(e for e in symbols if "sp500" not in e)

        # Variable initialization
        self.raw_data = dict()
        self.clean_data = dict()
        self.transformed_data = dict()

        # Load ^GSPC
        symbol = '^GSPC'
        gspc_path = self.data_dir.joinpath(f'{symbol}.csv')
        self.raw_data[symbol] =\
            pd.read_csv(str(gspc_path), index_col='Date')

        # Rename to get column names consistent
        self.raw_data[symbol] = self.raw_data[symbol].rename(
            index=str, columns={'Adj Close': 'Adj. Close'})

        # Cast to dates
        self.raw_data[symbol].index = \
            pd.to_datetime(self.raw_data[symbol].index)

        # Get dates
        self.dates = self.raw_data[symbol].index

        self.data_cleaning(symbol)

    def load_data(self, symbol):
        """
        Loads data

        The raw data is stored in self.raw_data.
        The cleaned data is stored in self.clean_data.

        Parameters
        ----------
        symbol : str
            The ticker symbol to load.
        """

        if symbol not in self.available_symbols:
            msg = f'{symbol} not available.\nAvailable symbols:\n'
            for s in self.available_symbols:
                msg += f'{s}\n'
            raise RuntimeError(msg)

        if symbol not in self.raw_data.keys():
            path = self.data_dir.joinpath(f'{symbol}.csv')
            self.raw_data[symbol] = pd.read_csv(str(path),
                                                parse_dates=True,
                                                index_col='Date')

        self.data_cleaning(symbol)

    def data_cleaning(self, symbol):
        """
        Cleans the data.

        * Make sure that the date is present in ^GSPC
        * Fill nans in a proper way

        Parameters
        ----------
        symbol : str
            The ticker symbol to clean.
        """

        if symbol not in self.raw_data.keys():
            raise RuntimeError(f'{symbol} not found in self.clean_data.')

        intersect = self.dates.intersection(self.raw_data[symbol].index)

        self.clean_data[symbol] = self.raw_data[symbol].loc[intersect].copy()
        self.clean_data[symbol].fillna(method='ffill', inplace=True)
        self.clean_data[symbol].fillna(method='bfill', inplace=True)

    def transform(self, func, *args, **kwargs):
        """
        Apply a transformation to all data frames of the clean data.

        The transformed_data dict will be populated on the first call.

        Parameters
        ----------
        func : function
            A function acting on a DataFrame. The function must take a
            DataFrame as input, and work on it inplace.
        args : tuple
            Positional arguments for the input function.
        kwargs : dict
            Keyword arguments for the input function.
        """

        # Initialize on first call
        if not self.transformed_data:
            for key in self.clean_data.keys():
                self.transformed_data[key] = self.clean_data[key].copy()

        # Append possible missing keys
        if len(self.transformed_data.keys()) != len(self.clean_data.keys()):
            msg = ('New symbols added after first transformation. '
                   'Reset with self.reset_transform and call the '
                   'transformations again.')
            raise RuntimeError(msg)

        for key in self.transformed_data.keys():
            func(self.transformed_data[key], *args, **kwargs)

    def reset_transform(self):
        """
        Resets all transformations by setting the self.transformed_data to an
        empty dict.
        """

        self.transformed_data = dict()

    def plot(self, features):
        """
        Plots the clean data of the selected features.

        Parameters
        ----------
        features : list
            List of the features to be plotted

        Returns
        -------
        ax : matplotlib.AxesSubplot
            The axes of the plot
        """

        symbol = '^GSPC'
        plot_frame = self.clean_data[symbol].loc[:, features]
        col_names = {f: f'{symbol} {f}' for f in features}
        plot_frame.rename(index=str, columns=col_names, inplace=True)

        symbols = [k for k in self.clean_data.keys() if k != '^GSPC']
        for symbol in symbols:
            plot_frame = \
                pd.merge(plot_frame,
                         self.clean_data[symbol].loc[:, features],
                         left_index=True, right_index=True)
            col_names = {f: f'{symbol} {f}' for f in features}
            plot_frame.rename(index=str, columns=col_names, inplace=True)

        # Cast to dates
        plot_frame.index = pd.to_datetime(plot_frame.index)

        ax = plot_frame.plot()
        _ = ax.set_ylabel('USD')
        ax.grid()

        return ax
