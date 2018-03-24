#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators import latest_day


class TestLatestDay(unittest.TestCase):
    def setUp(self):
        row = np.array(range(1, 6))
        self.last_row = row*100
        self.x = np.array([row,
                           row*10,
                           self.last_row])

    def test___init__(self):
        reg = latest_day.LatestDay(42)
        expected = 42
        self.assertEqual(expected, reg.days)

    def test_fit(self):
        reg = latest_day.LatestDay()
        reg.fit(self.x)
        self.assertTrue(np.allclose(self.last_row, reg.prediction_values))

    def test_predict(self):
        reg = latest_day.LatestDay()
        reg.fit(self.x)
        self.assertRaises(ValueError, reg.predict, self.x[:, :-1])

        expected = np.array(list(self.last_row)*reg.days).\
            reshape((reg.days, len(self.last_row)))

        self.assertTrue(np.allclose(expected, reg.predict(self.x)))


if __name__ == '__main__':
    unittest.main()