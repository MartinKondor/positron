import unittest
import positron.loss as loss
import numpy as np


class TestSum(unittest.TestCase):

    def test_dcross_entropy(self):
        a = np.array([np.nan, 0., np.inf, -np.inf, 100, 100])
        y = np.array([np.nan, 0., np.inf, -np.inf, 100, 0])
        o = np.array([0., 0., 0., 0., 0, 100])
        self.assertEqual(loss.dcross_entropy(a, y), o, "Should replace nan,inf,-inf with 0")


if __name__ == '__main__':
    unittest.main()
