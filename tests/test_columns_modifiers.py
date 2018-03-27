#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from utils.column_modifiers import feature_generator
from utils.column_modifiers import target_generator
from utils.column_modifiers import keep_columns


class TestColumnsGenerator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(np.array([[1, 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9],
                                         [10, 11, 12]]),
                               columns=['a', 'b', 'c'],
                               dtype=float)

    def test_feature_generator(self):
        expected = pd.DataFrame(
            np.array([[1, 2, 3, 6.0, 9.0],
                      [4, 5, 6, 9.0, 12.0],
                      [7, 8, 9, 12.0, np.nan],
                      [10, 11, 12, np.nan, np.nan]]),
            columns=['a', 'b', 'c', 'c + 1 days', 'c + 2 days'])
        result = target_generator(self.df, 'c', [1, 2])

        self.assertTrue(expected.equals(result))

    def test_target_generator(self):
        expected = pd.DataFrame(
            np.array([[1, np.nan, np.nan, np.nan, 2, 3],
                      [4, np.nan, np.nan, 2.0, 5, 6],
                      [7, np.nan, 2.0, 5.0, 8, 9],
                      [10, 2.0, 5.0, 8.0, 11, 12]]),
            columns=['a', 'b - 3 days', 'b - 2 days', 'b - 1 days', 'b', 'c'])

        result = feature_generator(self.df, 'b', 3)

        self.assertTrue(expected.equals(result))

    def test_keep_columns(self):
        expected = pd.DataFrame(
            np.array([[1, 3],
                      [4, 6],
                      [7, 9],
                      [10, 12]]),
            columns=['a', 'c'],
            dtype=float)

        result = keep_columns(self.df, ['a', 'c'])

        self.assertTrue(expected.equals(result))


if __name__ == '__main__':
    unittest.main()
