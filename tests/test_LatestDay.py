#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators import latest_day


class TestLatestDay(unittest.TestCase):
    def setUp(self):
        end = 4
        row = np.array(range(1, end))
        self.x = np.array([row,
                           row+end-1,
                           row+2*end-1])
        self.y = np.array(row+3*end-1)

    def test_fit(self):
        reg = latest_day.LatestDay()
        reg.fit(self.x, self.y)
        self.assertTrue(np.allclose(self.y[-1], reg.prediction_values))

    def test_predict(self):
        reg = latest_day.LatestDay()
        reg.fit(self.x, self.y)
        self.assertRaises(ValueError, reg.predict, self.x[:, :-1])

        expected = np.repeat(reg.prediction_values.copy()[np.newaxis, :],
                             len(self.y),
                             axis=0)

        self.assertTrue(np.allclose(expected, reg.predict(self.x)))


if __name__ == '__main__':
    unittest.main()