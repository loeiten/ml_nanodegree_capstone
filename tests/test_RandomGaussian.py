#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np
from estimators import random_gaussian


class TestRandomGaussian(unittest.TestCase):
    def setUp(self):
        row = np.array(range(1, 6))
        self.x = np.array([row,
                           row * 10,
                           row * 100])

    def test___init__(self):
        reg = random_gaussian.RandomGaussian(days=42)
        expected = 42
        self.assertEqual(expected, reg.days)

    def test_fit(self):
        reg = random_gaussian.RandomGaussian()
        reg.fit(self.x)
        expected = np.array([37., 74., 111., 148., 185.])
        self.assertTrue(np.allclose(expected, reg.mean))
        expected = np.array([44.69899328, 89.39798655, 134.09697983,
                             178.79597311, 223.49496639])
        self.assertTrue(np.allclose(expected, reg.std))

    def test_predict(self):
        reg = random_gaussian.RandomGaussian(seed=42)
        reg.fit(self.x)
        self.assertRaises(ValueError, reg.predict, self.x[:, :-1])

        expected = np.array([[59.20262259, 61.63944986, 197.85307683,
                              420.31160525, 132.66789939],
                             [26.53431374, 215.17844605, 213.9106794,
                              64.05987032, 306.2594387],
                             [16.28569566, 32.36469775, 143.44640985,
                              -194.08680317, -200.510453],
                             [11.86631351, -16.54506288, 153.13961822,
                              -14.35104819, -130.64276826],
                             [102.51302447, 53.81605332, 120.0553283,
                              -106.73923839, 63.33320128],
                             [41.95812809, -28.89650836, 161.37996959,
                              40.60822095, 119.80791519],
                             [10.10432019, 239.58994023, 109.19006293,
                              -41.11445481, 368.83464748]])

        self.assertTrue(np.allclose(expected, reg.predict(self.x)))


if __name__ == '__main__':
    unittest.main()
