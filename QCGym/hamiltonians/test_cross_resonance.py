import unittest
from QCGym.hamiltonians.cross_resonance import CrossResonance
import numpy as np


class TestCrossRes(unittest.TestCase):

    def test_shape(self):
        H = CrossResonance()
        inp = [([1, 2], [10, 20]), ([4, 5], [40, 50]),
               ([7, 8], [70, 80]), ([1, 4], [10, 40])]
        self.assertEqual(H(inp).shape, (40, 4, 4))


if __name__ == '__main__':
    unittest.main()
