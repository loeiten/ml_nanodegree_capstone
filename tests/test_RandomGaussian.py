#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators import random_gaussian


class TestRandomGaussian(unittest.TestCase):
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
        reg = random_gaussian.RandomGaussian()
        reg.fit(self.x, self.y)
        expected = 14.0
        self.assertEqual(expected, reg.mean)
        expected = 4.242640687119285
        self.assertEqual(expected, reg.std)

    def test_predict(self):
        reg = random_gaussian.RandomGaussian(seed=42)
        reg.fit(self.x, self.y)
        self.assertRaises(ValueError, reg.predict, self.x[:, :-1])

        expected = np.array([[16.10737968],
                             [13.41339425],
                             [16.74790974],
                             [20.46166844],
                             [13.00657137],
                             [13.00664102],
                             [20.70003254]])

        result = reg.predict(self.x)

        self.assertTrue(np.allclose(expected, result))


if __name__ == '__main__':
    unittest.main()
