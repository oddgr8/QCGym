from QCGym.hamiltonians.generic_hamiltonian import GenericHamiltonian
from QCGym.interpolators.identity import IdentityInterpolator
import numpy as np
from math import cos, sin
import logging
logger = logging.getLogger(__name__)


class CrossResonance(GenericHamiltonian):
    """
    Hamiltonian for Cross Resonance qubits.

    Parameters
    ----------
        smoothing : Interpolator
            How to interpolate parameters inside timesteps
        num_qubits : int
            Number of qubits we are dealing with
    """

    def __init__(self, smoothing=IdentityInterpolator(10), num_qubits=2):
        self.smoothing = smoothing
        self.num_qubits = num_qubits

    def hamil_eval(self, params=[1,1,1,0,1]):
        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        [sigma, wxx, w1,w2, phi] = params
        H = (sigma*wxx/(4*(w1-w2)))*(cos(phi)*np.kron(pauli_z, pauli_x) + sin(phi)*np.kron(pauli_z, pauli_y))
        return H

    def __call__(self, control_params):
        '''
        Returns Hamiltonian over time

        Parameters
        ----------
            control_params : ndarray of shape(num_params, timesteps)
                Control Parameters from which to generate Hamiltonians
            num_qubits : int
                Number of qubits we are dealing with

        Returns
        -------
            hamiltonians : ndarray of shape(time_steps*mesh, ...shape_of_hamiltonian)
                Interpolated Hamiltonians
        '''
        control_params = np.array(
            [self.smoothing(x) for x in control_params]).T  # Here it is (steps, num_params)
        return np.array([self.hamil_eval(x) for x in control_params])

    def __str__(self):
        return f"Cross Res. w/{self.smoothing}"
