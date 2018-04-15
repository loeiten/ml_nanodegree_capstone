#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from estimators.predictions import calculate_rolling_prediction
from estimators.predictions import calculate_normal_prediction
from utils.column_modifiers import target_generator


class TestPredictions(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(np.array(range(16)), columns=['x'])
        df = target_generator(df, 'x', [2, 3])
        x = df.loc[:, ['x']]
        y = df.loc[:, ['x + 2 days', 'x + 3 days']]
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, shuffle=False, test_size=12)

        # Obtain the day of prediction
        # I.e. for a column named x + 2 days, we would expect the two last rows
        # to contain nan
        self.prediction_days = self.y_test.isnull().sum()

    def test_calculate_rolling_prediction(self):
        expected = pd.DataFrame(np.array(
            [[6., 7.],
             [7., 8.],
             [8., 9.],
             [9., 10.],
             [10., 11.],
             [11., 12.],
             [12., 13.],
             [13., 14.],
             [14., 15.],
             [15., np.nan]]),
            columns=['x + 2 days', 'x + 3 days'],
            index=range(4, 14))

        reg = linear_model.LinearRegression()

        result = \
            calculate_rolling_prediction(reg,
                                         self.x_train,
                                         self.x_test,
                                         self.y_train,
                                         self.y_test,
                                         self.prediction_days)

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))

        result, train_result = \
            calculate_rolling_prediction(reg,
                                         self.x_train,
                                         self.x_test,
                                         self.y_train,
                                         self.y_test,
                                         self.prediction_days,
                                         training_prediction=True)

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))

        train_expected = pd.DataFrame(np.array(
            [[5., 6.],
             [6., 7.],
             [7., 8.],
             [8., 9.],
             [9., 10.],
             [10., 11.],
             [11., 12.],
             [12., 13.],
             [13., 14.],
             [14., np.nan]]),
            columns=['x + 2 days fit prediction', 'x + 3 days fit prediction'],
            index=range(3, 13))

        self.assertTrue(np.allclose(train_result.values,
                                    train_expected.values,
                                    equal_nan=True))

    def test_calculate_normal_prediction(self):
        expected = pd.DataFrame(np.array(
            [[6., 7.],
             [7., 8.],
             [8., 9.],
             [9., 10.],
             [10., 11.],
             [11., 12.],
             [12., 13.],
             [13., 14.],
             [14., 15.],
             [15., np.nan]]),
            columns=['x + 2 days', 'x + 3 days'],
            index=range(5, 15))

        reg = linear_model.LinearRegression()

        result = \
            calculate_normal_prediction(reg,
                                        self.x_train,
                                        self.x_test,
                                        self.y_train,
                                        self.y_test)

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))

    def test_same_results(self):

        reg = linear_model.LinearRegression()

        expected = \
            calculate_normal_prediction(reg,
                                        self.x_train,
                                        self.x_test,
                                        self.y_train,
                                        self.y_test)
        result = \
            calculate_rolling_prediction(reg,
                                         self.x_train,
                                         self.x_test,
                                         self.y_train,
                                         self.y_test,
                                         self.prediction_days)

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))


if __name__ == '__main__':
    unittest.main()
