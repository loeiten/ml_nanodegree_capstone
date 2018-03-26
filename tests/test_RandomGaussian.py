#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators import random_gaussian


class TestRandomGaussian(unittest.TestCase):
    def setUp(self):
        end = 4
        row = np.array(range(1, end))
        self.x = np.array([row,
                           row+end-1,
                           row+2*end-1])
        self.y = np.array(row+3*end-1)

    def test_fit(self):
        reg = random_gaussian.RandomGaussian()
        reg.fit(self.x, self.y)
        expected = 13
        self.assertEqual(expected, reg.mean)
        expected = 0.816496580927726
        self.assertEqual(expected, reg.std)

    def test_predict(self):
        reg = random_gaussian.RandomGaussian(seed=42)
        reg.fit(self.x, self.y)
        self.assertRaises(ValueError, reg.predict, self.x[:, :-1])

        expected = np.array([[13.40556541],
                             [12.88710767],
                             [13.52883548]])

        self.assertTrue(np.allclose(expected, reg.predict(self.x)))


if __name__ == '__main__':
    unittest.main()
