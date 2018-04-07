#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from utils.scorers import normalized_root_mean_square_error


class TestScorers(unittest.TestCase):
    def test_normalized_root_mean_square_error(self):
        expected = 1.0
        true = pd.DataFrame(np.array([[1, 1], [2, 2], [np.nan, np.nan]]))
        pred = pd.DataFrame(np.array([[np.nan, np.nan], [1, 1], [2, 2]]))

        result = normalized_root_mean_square_error(true, pred)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
