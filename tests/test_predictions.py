#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from estimators.predictions import calculate_rolling_prediction
from utils.column_modifiers import target_generator


class TestPredictions(unittest.TestCase):
    def test_calculate_rolling_prediction(self):
        expected = pd.DataFrame(np.array(
            [[5., np.nan],
             [6.,  6.],
             [7., 7.],
             [8., 8.],
             [9., 9.],
             [10., 10.],
             [11., 11.],
             [12., 12.],
             [13., np.nan]]),
            columns=['x + 2 days', 'x + 3 days'],
            index=range(5, 14))

        df = pd.DataFrame(np.array(range(16)), columns=['x'])
        df = target_generator(df, 'x', [2, 3])
        x = df.loc[:, ['x']]
        y = df.loc[:, ['x + 2 days', 'x + 3 days']]
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, shuffle=False, test_size=12)
        reg = linear_model.LinearRegression()

        result = \
            calculate_rolling_prediction(reg, x_train, x_test, y_train, y_test)

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))


if __name__ == '__main__':
    unittest.main()
