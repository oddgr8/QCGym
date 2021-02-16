from QCGym.hamiltonians.cross_resonance import CrossResonance
from QCGym.fidelities.trace_fidelity import TraceFidelity
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
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

    def __init__(self, max_timesteps=30, target=CNOT, hamiltonian=CrossResonance(), fidelity=TraceFidelity(), dt=0.1):
        self.max_timesteps = max_timesteps
        self.target = target
        self.hamiltonian = hamiltonian
        self.fidelity = fidelity
        self.dt = dt

        logger.info(
            f"GenEnv-{max_timesteps}-{CNOT}-{hamiltonian}-{fidelity}-{dt}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4, 4))
        self.action_space = spaces.Tuple((spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # Capital Omega1
                                          spaces.Box(
                                              low=-np.inf, high=np.inf, shape=(1,)),  # Capital Omega2
                                          spaces.Box(
                                              low=-np.inf, high=np.inf, shape=(1,)),  # Small omega1
                                          spaces.Box(
                                              low=-np.inf, high=np.inf, shape=(1,)),  # Small omega2
                                          spaces.Box(
                                              low=-np.pi, high=np.pi, shape=(1,)),  # phi1
                                          spaces.Box(
                                              low=-np.pi, high=np.pi, shape=(1,)),  # phi2
                                          ))

        self.actions_so_far = []

    def is_hermitian(self, H):
        return np.all(H == np.conjugate(H).T)

    def step(self, action):
        """
        Take one action for one timestep

        Parameters
        ----------
            action : 6-touble of doubles
                (Omega1, Omega2, omega1, omega2, phi1, phi2)

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
        self.actions_so_far.append(action)
        logger.info(f"Action#{len(self.actions_so_far)}={action}")

        if len(self.actions_so_far) == self.max_timesteps:
            H = np.sum(self.hamiltonian(
                np.array(self.actions_so_far).T), axis=0)

            if not self.is_hermitian(H):
                logger.error(f"{H} is not")

            U = expm(-1j*self.dt*H/H_CROSS)

            if not np.allclose(np.matmul(U, np.conjugate(U.T)), np.eye(4)):
                logger.error(
                    f"Unitary Invalid-Difference is{np.matmul(U,U.T)-np.eye(4)}")
            if not np.isclose(np.abs(np.linalg.det(U)), 1):
                logger.error(f"Det Invalid-{np.abs(np.linalg.det(U))}")

            return len(self.actions_so_far), self.fidelity(U, self.target), True, {}

        return len(self.actions_so_far), 0, False, {}

    def reset(self):
        self.actions_so_far = []
        logger.info("GenEnv Reset")

    def render(self, mode='human'):
        pass

    def close(self):
        pass
