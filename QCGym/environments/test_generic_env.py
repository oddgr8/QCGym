import unittest
from QCGym.environments.generic_env import GenericEnv
import numpy as np
from gym import spaces


class TestGenericEnv(unittest.TestCase):

    def test_env(self):
        E = GenericEnv()
        self.assertEqual(E.observation_space, spaces.Discrete(1000))
        self.assertEqual(E.reward_range, (float(0), float(1)))

        for i in range(29):
            obs, rew, done, _ = E.step(np.random.rand(2, 2))
            self.assertEqual(obs, i+1)
            self.assertEqual(rew, 0)
            self.assertFalse(done)

        obs, rew, done, _ = E.step(np.random.rand(2, 2))

        self.assertEqual(obs, 30)
        self.assertNotEqual(rew, 0)
        self.assertTrue(done)

        E.reset()

        for i in range(29):
            obs, rew, done, _ = E.step(np.random.rand(2, 2))
            self.assertEqual(obs, i+1)
            self.assertEqual(rew, 0)
            self.assertFalse(done)

        obs, rew, done, _ = E.step(np.random.rand(2, 2))

        self.assertEqual(obs, 30)
        self.assertNotEqual(rew, 0)
        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
