#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from estimators.lstm import prepare_input
from estimators.lstm import make_model
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
        # Single target
        expected_x = np.array([[[0., 1., 2.],
                                [3., 4., 5.]],
                               [[3., 4., 5.],
                                [6., 7., 8.]],
                               [[6., 7., 8.],
                                [9., 10., 11.]],
                               [[9., 10., 11.],
                                [12., 13., 14.]]])
        expected_y = np.array([[11.], [14.], [17.], [20.]])
        y = self.y[:, np.newaxis]
        x, y = prepare_input(self.x, y, self.time_step)
        self.assertTrue(np.allclose(expected_x, x))
        self.assertTrue(np.allclose(expected_y, y, equal_nan=True))

        # Multiple target
        expected_y = np.array([[11., 11., 11.],
                               [14., 14., 14.],
                               [17., 17., 17.],
                               [20., 20., 20.]])
        y = self.y[:, np.newaxis].repeat(3, axis=1)
        x, y = prepare_input(self.x, y, self.time_step)
        self.assertTrue(np.allclose(expected_x, x))
        self.assertTrue(np.allclose(expected_y, y, equal_nan=True))

    def test_make_model(self):
        shape = (2, 3, 4)
        targets = 3
        drop_out = 0.0
        recurrent_drop_out = 0.0
        loss = 'mse'
        optimizer = 'adam'

        model = make_model(cells=[10, 20, 30],
                           input_shape=shape,
                           targets=targets,
                           drop_out=drop_out,
                           recurrent_drop_out=recurrent_drop_out,
                           loss=loss,
                           optimizer=optimizer)
        self.assertEqual(model.count_params(), 9293)

        model = make_model(cells=[10, 20],
                           input_shape=shape,
                           targets=targets,
                           drop_out=drop_out,
                           recurrent_drop_out=recurrent_drop_out,
                           loss=loss,
                           optimizer=optimizer)
        self.assertEqual(model.count_params(), 3143)

        model = make_model(cells=[10],
                           input_shape=shape,
                           targets=targets,
                           drop_out=drop_out,
                           recurrent_drop_out=recurrent_drop_out,
                           loss=loss,
                           optimizer=optimizer)
        self.assertEqual(model.count_params(), 633)

    def test_fit(self):
        reg = lstm.LSTMRegressor(epochs=1)
        reg.fit(self.x, self.y)
        self.assertTrue(type(reg), Sequential)

    def test_predict(self):
        reg = lstm.LSTMRegressor(seed=42, epochs=1)
        self.assertRaises(RuntimeError, reg.predict, self.x)

        reg.fit(self.x, self.y)

        result = reg.predict(self.x)

        # Hard to assert absolute equality across platforms due to the
        # randomness in keras
        self.assertEqual((~np.isnan(result)).shape, (7, 1))


if __name__ == '__main__':
    unittest.main()
