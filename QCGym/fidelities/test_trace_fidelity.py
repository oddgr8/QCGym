import unittest
from QCGym.fidelities.trace_fidelity import TraceFidelity
import numpy as np


class TestTraceFidelity(unittest.TestCase):

    def test_fidelity(self):
        F = TraceFidelity()
        m1 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4]])
        m2 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4]])
        self.assertEqual(F(m1, m2), 36)

        m1 = np.eye(4)
        m2 = np.eye(4)
        self.assertEqual(F(m1, m2), 1)


if __name__ == '__main__':
    unittest.main()
