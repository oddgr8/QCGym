from QCGym.hamiltonians.generic_hamiltonian import GenericHamiltonian
from QCGym.interpolators.int_identity import InteractiveIdentityInterpolator
import numpy as np
from gym import spaces
import logging
logger = logging.getLogger(__name__)


class InteractiveCrossResonance(GenericHamiltonian):
    """
    Hamiltonian for Cross Resonance qubits.

    Parameters
    ----------
        smoothing : Interpolator
            How to interpolate parameters inside timesteps
        num_qubits : int
            Number of qubits we are dealing with
        dt : double
            (Time of each step)/mesh_size
    """

    def __init__(self, smoothing=InteractiveIdentityInterpolator(10), num_qubits=2, dt=0.1):
        self.smoothing = smoothing
        self.num_qubits = num_qubits
        self.dt = dt

        self.action_space = spaces.Tuple((spaces.Box(low=-500, high=500, shape=(2,)),  # Omega
                                          spaces.Box(
                                              low=-100*np.pi, high=100*np.pi, shape=(2,)),  # phi
                                          ))

        self.omega1 = 1
        self.omega2 = 2
        self.omega1rf = self.omega2
        self.omega2rf = self.omega1
        self.omegaxx = 1.5
        pauli_x = np.array([[0, 1], [1, 0]])
        # pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        I = np.eye(2)
        self.sigma1z = np.kron(pauli_z, I)
        self.sigma1x = np.kron(pauli_x, I)
        self.sigma2z = np.kron(I, pauli_z)
        self.sigma2x = np.kron(I, pauli_x)
        self.sigmaxx = np.kron(pauli_x, pauli_x)
        self.steps_so_far = 0  # Used by Hamil_eval

    def hamil_eval(self, params):
        Omega1 = params[0, 0]
        Omega2 = params[0, 1]
        phi1 = params[1, 0]
        phi2 = params[1, 1]
        t = self.steps_so_far*self.dt

        H = (0.5*self.omega1*self.sigma1z) + \
            (Omega1*np.cos(self.omega1rf*t + phi1)*self.sigma1x) + \
            (0.5*self.omega2*self.sigma2z) + \
            (Omega2*np.cos(self.omega2rf*t+phi2)*self.sigma2x) + \
            (0.5*self.omegaxx*self.sigmaxx)

        if not np.all(H == np.conjugate(H).T):
            logger.error(f"{H} is not hermitian with params {params}")
        self.steps_so_far += 1
        return H

    def __call__(self, control_param, done=False):
        '''
        Returns Hamiltonians over time

        Parameters
        ----------
            control_params : iterable of shape of action_space
                Control Parameters from which to generate Hamiltonians

        Returns
        -------
            hamiltonians : ndarray of shape(time_steps*mesh, ...shape_of_hamiltonian)
                Interpolated Hamiltonians
        '''

        return np.array([self.hamil_eval(x) for x in self.smoothing(control_param, done=done)])

    def __str__(self):
        return f"Cross Res. w/{self.smoothing}"

    def reset(self):
        self.steps_so_far = 0
