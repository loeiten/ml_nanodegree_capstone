#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from data_preparation.ochlva_data import OCHLVAData


class TestOCHLVAData(unittest.TestCase):
    def setUp(self):
        self.data = OCHLVAData()

    def test___init__(self):
        self.assertTrue('^GSPC' in self.data.raw_data.keys())
        self.assertTrue('^GSPC' in self.data.clean_data.keys())
        self.assertTrue(self.data.clean_data['^GSPC'] is not None)

    def test_load_data(self):
        self.assertRaises(RuntimeError, self.data.load_data, 'foo_bar_baz')
        self.data.load_data('AAPL')
        self.assertTrue('AAPL' in self.data.raw_data.keys())
        self.assertTrue('AAPL' in self.data.clean_data.keys())
        self.assertTrue(self.data.clean_data['AAPL'] is not None)
        self.assertFalse(self.data.clean_data['AAPL'].empty)

    def test_transform(self):
        self.data.transform(func)
        self.assertTrue('Close' in self.data.clean_data['^GSPC'].columns)
        self.assertTrue('Close' not in
                        self.data.transformed_data['^GSPC'].columns)
        self.data.load_data('AAPL')
        self.assertRaises(RuntimeError, self.data.transform, func)

    def test_reset_transform(self):
        self.data.transform(func)
        self.assertTrue(self.data.transformed_data)
        self.data.reset_transform()
        self.assertFalse(self.data.transformed_data)

    def test_plot(self):
        self.data.load_data('AAPL')
        ax = self.data.plot(['Close', 'Adj. Close'])
        lines = ax.get_lines()
        self.assertTrue(len(lines) == 4)
        for l in lines:
            self.assertTrue(len(l.get_ydata()) > 0)


def func(df):
    df.drop('Close', axis=1, inplace=True)


if __name__ == '__main__':
    unittest.main()
