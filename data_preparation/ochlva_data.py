#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


class OCHLVAData(object):
    """
    Class treating Open, Close, High, Low, Volume and Adjusted close
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

        # Load ^GSPC
        gspc_path = self.data_dir.joinpath(f'^GSPC.csv')
        self.raw_data['^GSPC'] = pd.read_csv(str(gspc_path), index_col='Date')

        # Get dates
        self.dates = self.raw_data['^GSPC'].index

        self.data_cleaning('^GSPC')

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
            self.raw_data[symbol] = pd.read_csv(str(path), index_col='Date')

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
