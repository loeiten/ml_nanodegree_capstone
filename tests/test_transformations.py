#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from utils.transformations import StockMinMax
from utils.transformations import subtract_previous_row


class TestTransformations(unittest.TestCase):
    def test_subtract_previous_row(self):

        df = pd.DataFrame(np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9],
                                    [10, 11, 12]]),
                          columns=['a', 'b', 'c'], dtype=float)

        expected = pd.DataFrame(np.array([[np.nan, 2, np.nan],
                                          [3, 5, 3],
                                          [3, 8, 3],
                                          [3, 11, 3]]),
                                columns=['a', 'b', 'c'], dtype=float)
        result = subtract_previous_row(df, ['a', 'c'])

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))


class TestStockMinMax(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.array([[1, 2, 3, 4],
                                         [5, 6, 7, 8],
                                         [9, 10, 11, 12]]),
                               columns=['a', 'a + 1', 'c', 'c - 1'],
                               dtype=float)

    def test_fit(self):
        scaler = StockMinMax()
        scaler.fit(self.df)
        clean_features = ['a', 'c']

        max_scaler =\
            pd.Series(self.df.loc[:, clean_features].values.max(axis=0),
                      index=['a', 'c'])
        self.assertTrue(scaler.max.equals(max_scaler))

        min_scaler =\
            pd.Series(self.df.loc[:, clean_features].values.min(axis=0),
                      index=['a', 'c'])
        self.assertTrue(scaler.min.equals(min_scaler))

    def test_transform(self):
        scaler = StockMinMax()
        scaler.fit(self.df)

        expected = pd.DataFrame(np.array([[0, 0.125, 0, 0.125],
                                          [0.5, 0.625, 0.5, 0.625],
                                          [1, 1.125, 1, 1.125]]),
                                columns=['a', 'a + 1', 'c', 'c - 1'],
                                dtype=float)

        x_prime = scaler.transform(self.df)

        self.assertTrue(x_prime.equals(expected))

    def test_inverse_transform(self):
        scaler = StockMinMax()
        scaler.fit(self.df)

        x = scaler.inverse_transform(scaler.transform(self.df))

        self.assertTrue(x.equals(self.df))


if __name__ == '__main__':
    unittest.main()
