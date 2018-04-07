#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators import latest_day


class TestLatestDay(unittest.TestCase):
    def setUp(self):
        # Cannot store nan in integer types
        # https://stackoverflow.com/questions/11548005/numpy-or-pandas-keeping-array-type-as-integer-while-having-a-nan-value
        self.x = np.array(range(21), dtype=float).reshape(7, 3)
        self.days = 2
        # Make target (the prediction) self.days days from today
        self.y = np.roll(self.x[:, -1], -self.days)
        # Set elements rolled beyond last position to nan
        self.y[-self.days:] = np.nan

    def test_fit(self):
        reg = latest_day.LatestDay()
        reg.fit(self.x, self.y)
        expected = self.x[-1, -1]
        self.assertTrue(np.allclose(expected, reg.prediction_values))

    def test_predict(self):
        reg = latest_day.LatestDay()
        reg.fit(self.x, self.y)
        self.assertRaises(ValueError, reg.predict, self.x[:, :-1])

        expected = \
            np.array([self.x[-1, -1]]*len(self.x)).reshape(len(self.x), 1)

        result = reg.predict(self.x)

        self.assertTrue(np.allclose(expected, result))


if __name__ == '__main__':
    unittest.main()
