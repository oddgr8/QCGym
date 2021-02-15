from QCGym.hamiltonians.cross_resonance import CrossResonance
from QCGym.fidelities.trace_fidelity import TraceFidelity
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import logging
logger = logging.getLogger(__name__)


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class GenericEnv(gym.Env):
    """
    Gym Environment for Quantum Control.

    Parameters
    ----------
        smoothing : Interpolator
            How to interpolate parameters inside timesteps
        num_qubits : int
            Number of qubits we are dealing with
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, target=CNOT, hamiltonian=CrossResonance(), fidelity=TraceFidelity()):
        pass

    def step(self, action):
        return None, None, None, None

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
