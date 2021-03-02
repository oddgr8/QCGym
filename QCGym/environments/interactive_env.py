from QCGym.hamiltonians.int_cross_resonance import InteractiveCrossResonance
from QCGym.fidelities.trace_fidelity import TraceFidelity
import gym
import numpy as np
from gym import spaces
from scipy.linalg import expm
import logging
logger = logging.getLogger(__name__)


CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

# We use natural units
H_CROSS = 1


class InteractiveEnv(gym.Env):
    """
    Interactive Gym Environment for Quantum Control.

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
    observation_space = spaces.Tuple((spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4)),
                                      spaces.Box(
                                          low=-np.inf, high=np.inf, shape=(4, 4)),
                                      spaces.Discrete(1000)))
    reward_range = (float(0), float(1))

    def __init__(self, max_timesteps=30, target=CNOT, hamiltonian=InteractiveCrossResonance(), fidelity=TraceFidelity(), dt=0.1):
        self.max_timesteps = max_timesteps
        self.target = target
        self.hamiltonian = hamiltonian
        self.fidelity = fidelity
        self.dt = dt
        self.U = np.eye(4)

        self.name = f"IntEnv-{max_timesteps}-{CNOT}-{hamiltonian}-{fidelity}-{dt}"

        logger.info(self.name)

        self.action_space = self.hamiltonian.action_space

        self.actions_so_far = []

    def __str__(self):
        return self.name

    def step(self, action):
        """
        Take one action for one timestep

        Parameters
        ----------
            action : ActionSpace
                Object in Hamiltonian.action_space

        Returns
        -------
            observation : ndarray(shape=(4,4)),ndarray(shape=(4,4)),int
                Real part of state, Imag part of state, Number of timesteps done
            reward : double
                Returns fidelity on final timestep, zero otherwise
            done : boolean
                Returns true of episode is finished
            info : dict
                Additional debugging Info
        """
        self.actions_so_far.append(action)
        logger.info(f"Action#{len(self.actions_so_far)}={action}")

        done = len(self.actions_so_far) == self.max_timesteps

        H = np.sum(self.hamiltonian(action, done=done), axis=0)

        if not np.all(H == np.conjugate(H).T):
            logger.error(
                f"{H} is not Hermitian with actions as {np.array(self.actions_so_far)}")

        self.U = expm(-1j*self.dt*H/H_CROSS) @ self.U

        if not np.allclose(np.matmul(self.U, np.conjugate(self.U.T)), np.eye(4)):
            logger.error(
                f"Unitary Invalid-Difference is{np.matmul(self.U,self.U.T)-np.eye(4)}")
        if not np.isclose(np.abs(np.linalg.det(self.U)), 1):
            logger.error(f"Det Invalid-{np.abs(np.linalg.det(self.U))}")

        if done:
            reward = self.fidelity(self.U, self.target)
        else:
            reward = 0

        return (np.real(self.U), np.imag(self.U), len(self.actions_so_far)), reward, done, {}

    def reset(self):
        self.actions_so_far = []
        self.U = np.eye(4)
        self.hamiltonian.reset()
        logger.info("IntEnv Reset")
        return (np.eye(4), np.zeros((4, 4)), 0)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
