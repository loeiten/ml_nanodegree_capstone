#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators.lstm import prepare_input
from estimators import lstm


class TestLSTM(unittest.TestCase):
    def setUp(self):
        # Cannot store nan in integer types
        # https://stackoverflow.com/questions/11548005/numpy-or-pandas-keeping-array-type-as-integer-while-having-a-nan-value
        self.x = np.array(range(21), dtype=float).reshape(7, 3)
        self.days = 2
        # Make target (the prediction) self.days days from today
        self.y = np.roll(self.x[:, -1], -self.days)
        # Set elements rolled beyond last position to nan
        self.y[-self.days:] = np.nan

        self.time_step = 2

    def test_prepare_input(self):
        expected_x = np.array([[[0., 1., 2.],
                                [3., 4., 5.]],
                               [[3., 4., 5.],
                                [6., 7., 8.]],
                               [[6., 7., 8.],
                                [9., 10., 11.]],
                               [[9., 10., 11.],
                                [12., 13., 14.]],
                               [[12., 13., 14.],
                                [15., 16., 17.]],
                               [[15., 16., 17.],
                                [18., 19., 20.]]])
        expected_y = np.array([[11.], [14.], [17.], [20.], [np.nan], [np.nan]])
        y = self.y[:, np.newaxis]
        x, y = prepare_input(self.x, y, self.time_step)
        self.assertTrue(np.allclose(expected_x, x))
        self.assertTrue(np.allclose(expected_y, y, equal_nan=True))

    def test_fit(self):
        self.fail()

    def test_predict(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
