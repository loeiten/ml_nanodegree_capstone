#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to obtain and save the 50 stocks with the highest weights in S&P500.
"""

import datetime
import quandl
import pandas as pd
from time import sleep
from pathlib import Path
from dateutil.relativedelta import relativedelta


def get_sp_50_highest_weights():
    """
    Saves the 50 stocks with the highest weights in S&P500.
    """

    # Waiting time in order for quandl not to throw LimitExceededError
    wait_minutes = 2

    # Check for file
    cur_dir = Path(__file__).absolute().parent
    weight_files = list(cur_dir.glob('*-sp500.csv'))

    if len(weight_files) != 1:
        msg = 'Did not find the correct file to read S&P500 weights from.'
        raise FileNotFoundError(msg)

    weight_file = weight_files[0]

    to_date = datetime.datetime.strptime(weight_file.name.split('-sp500')[0],
                                         '%Y_%m_%d')
    from_date = to_date - relativedelta(years=5)

    weights_df = pd.read_csv(str(weight_file))
    symbols = weights_df.loc[:50, "Symbol"]

    for symbol in symbols:
        print(f'Processing {symbol}...')
        symbol = symbol.replace('.', '_')
        symbol_path = cur_dir.joinpath(f'{symbol}.csv')
        if not symbol_path.is_file():
            df = quandl.get(f'WIKI/{symbol}', start_date=from_date,
                            end_date=to_date)
            df.to_csv(str(symbol_path))
            print(f'...{symbol} downloaded, waiting {wait_minutes} minutes')
            # Multiply with 61 rather than 60 in case the script is super fast
            sleep(wait_minutes*61)
        else:
            print('...symbol already downloaded')


if __name__ == '__main__':
    get_sp_50_highest_weights()
