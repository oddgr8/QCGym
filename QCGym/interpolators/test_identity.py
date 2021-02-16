import unittest
from QCGym.interpolators.identity import IdentityInterpolator
import numpy as np


class TestIdentity(unittest.TestCase):

    def test_params(self):
        I = IdentityInterpolator(3)
        self.assertTrue(np.all(I(np.array([1, 2, 3])) ==
                               np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])))


if __name__ == '__main__':
    unittest.main()
