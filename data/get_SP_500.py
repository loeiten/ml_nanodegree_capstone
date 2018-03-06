#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to obtain and save the S&P500 to a csv.
"""

import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


def get_sp_500():
    """
    Gets the S&P500 rank, company, symbol and weight and saves it to a csv.
    """
    url = 'https://www.slickcharts.com/sp500'
    html = requests.get(url).content
    sp_500_df = pd.read_html(html)[-1]
    sp_500_df.set_index('Rank', inplace=True)

    # Read the buttons
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.findChildren('table')[0]
    rows = table.findChildren('tr')

    symbols = list()
    for row in rows[1:]:
        cells = row.findChildren('td')
        symbols.append(str(cells[2]).split('value=')[1].split('"')[1])

    # Update the symbols
    sp_500_df['Symbol'] = symbols

    date = datetime.datetime.today().strftime('%Y_%m_%d')
    file_name = Path(__file__).absolute().parent.joinpath(f"{date}-sp500.csv")
    sp_500_df.to_csv(str(file_name))


if __name__ == '__main__':
    get_sp_500()
