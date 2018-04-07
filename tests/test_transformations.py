#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from utils.transformations import subtract_previous_row


class TestTransformations(unittest.TestCase):
    def test_subtract_previous_row(self):

        df = pd.DataFrame(np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9],
                                    [10, 11, 12]]),
                          columns=['a', 'b', 'c'], dtype=float)

        expected = pd.DataFrame(np.array([[np.nan, 2, np.nan],
                                          [3, 5, 3],
                                          [3, 8, 3],
                                          [3, 11, 3]]),
                                columns=['a', 'b', 'c'], dtype=float)
        result = subtract_previous_row(df, ['a', 'c'])

        self.assertTrue(np.allclose(result.values,
                                    expected.values,
                                    equal_nan=True))


if __name__ == '__main__':
    unittest.main()
