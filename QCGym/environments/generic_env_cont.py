from QCGym.hamiltonians.cross_resonance import CrossResonance
from QCGym.fidelities.trace_fidelity import TraceFidelity
import gym
import numpy as np
from random import random
from gym import spaces
from math import isnan, pi
from QCGym.util import pulse_plot
from scipy.linalg import expm
import logging
logger = logging.getLogger(__name__)


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

# We use natural units
H_CROSS = 1


class GenericEnv(gym.Env):
    """
    Gym Environment for Quantum Control.

    Parameters
    ----------
        max_timesteps : int
            How many nanoseconds to run for
        target : ndarray of shape(4,4)
            Target Unitary we want to acheive
        hamiltonian : Hamiltonian
            The Hamiltonian to be used
        fidelity : Fidelity
            Fidelity to be used to calculate reward
        dt : double
            Timestep_size/mesh
    """
    metadata = {'render.modes': ['human']}
    observation_space = spaces.Discrete(1000)
    reward_range = (float(0), float(1))

    def __init__(self, max_timesteps=30, target=CNOT, hamiltonian=CrossResonance(), fidelity=TraceFidelity(), dt=0.1):
        self.max_timesteps = max_timesteps
        self.target = target
        self.hamiltonian = hamiltonian
        self.fidelity = fidelity
        self.dt = dt
        self.pulse_seq_best = []
        self.max_fidelity = 0

        logger.info(
            f"GenEnv-{max_timesteps}-{CNOT}-{hamiltonian}-{fidelity}-{dt}")

        self.action_space = self.hamiltonian.action_space

        self.actions_so_far = []

    def step(self, action):
        """
        Take one action for one timestep

        Parameters
        ----------
            action : ActionSpace
                Object in Hamiltonian.action_space

        Returns
        -------
            observation : int
                Number of timesteps done
            reward : double
                Returns fidelity on final timestep, zero otherwise
            done : boolean
                Returns true of episode is finished
            info : dict
                Additional debugging Info
        """
        if isnan(action[0][0]):
            action = [[10*(random()-0.5), 10*(random()-0.5)], [2*pi*(random()-0.5), 2*pi*(random()-0.5)]]

        self.actions_so_far.append(action)
        logger.info(f"Action#{len(self.actions_so_far)}={action}")

        if len(self.actions_so_far) == self.max_timesteps:
            H = np.sum(self.hamiltonian(self.actions_so_far), axis=0)

            if not np.all(H == np.conjugate(H).T):
                logger.error(
                    f"{H} is not Hermitian with actions as {np.array(self.actions_so_far)}")

            U = expm(-1j*self.dt*H/H_CROSS)

            if not np.allclose(np.matmul(U, np.conjugate(U.T)), np.eye(4)):
                logger.error(
                    f"Unitary Invalid-Difference is{np.matmul(U,U.T)-np.eye(4)}")
            if not np.isclose(np.abs(np.linalg.det(U)), 1):
                logger.error(f"Det Invalid-{np.abs(np.linalg.det(U))}")
            
            curr_fidelity = self.fidelity(U, self.target)

            if curr_fidelity > self.max_fidelity:
                self.max_fidelity = curr_fidelity
                self.action_seq_best = self.actions_so_far

            return len(self.actions_so_far), curr_fidelity, True, {}

        return len(self.actions_so_far), 0, False, {}

    def reset(self):
        self.actions_so_far = []
        logger.info("GenEnv Reset")
        return 0

    def render(self, mode='human'):
        pass

    def close(self):
        pulse_plot(self.pulse_seq_best, self.max_fidelity)
