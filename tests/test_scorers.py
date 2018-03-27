#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from scorers.scorers import nrmse


class TestScorers(unittest.TestCase):
    def test_nrmse(self):
        expected = 1.0
        true = pd.DataFrame(np.array([[1, 1], [2, 2], [np.nan, np.nan]]))
        pred = pd.DataFrame(np.array([[np.nan, np.nan], [1, 1], [2, 2]]))

        result = nrmse(true, pred)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
