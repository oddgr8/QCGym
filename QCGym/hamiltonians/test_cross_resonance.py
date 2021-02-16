import unittest
from QCGym.hamiltonians.cross_resonance import CrossResonance
import numpy as np


class TestCrossRes(unittest.TestCase):

    def test_shape(self):
        H = CrossResonance()
        input = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        self.assertEqual(H(input).shape, (20, 4, 4))


if __name__ == '__main__':
    unittest.main()
