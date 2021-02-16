import unittest
from QCGym.environments.generic_env import GenericEnv
import numpy as np


class TestGenericEnv(unittest.TestCase):

    def test_env(self):
        E = GenericEnv()
        for i in range(29):
            obs, rew, done, info = E.step(np.random.rand(6))
            self.assertEqual(obs, i+1)
            self.assertEqual(rew, 0)
            self.assertFalse(done)

        obs, rew, done, info = E.step(np.random.rand(6))

        self.assertEqual(obs, 30)
        self.assertNotEqual(rew, 0)
        self.assertTrue(done)

        E.reset()

        for i in range(29):
            obs, rew, done, info = E.step(np.random.rand(6))
            self.assertEqual(obs, i+1)
            self.assertEqual(rew, 0)
            self.assertFalse(done)

        obs, rew, done, info = E.step(np.random.rand(6))

        self.assertEqual(obs, 30)
        self.assertNotEqual(rew, 0)
        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
