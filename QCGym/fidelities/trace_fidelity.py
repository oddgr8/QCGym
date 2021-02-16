from QCGym.fidelities.generic_fidelity import GenericFidelity
import numpy as np


class TraceFidelity(GenericFidelity):
    """
    Trace fidelity among unitaries

    F = trace(unitaries, target)/comp_dim
    """

    def __call__(self, unitary, target):
        '''
        Returns Fidelity

            Parameters:
                unitary : square ndarray
                target : ndarray with same dimensions as unitary

            Returns:
                fidelity : double
                    |Tr(unitary.T@,target)/dim|
        '''
        return np.abs(np.trace(np.matmul(unitary.T, target)))/unitary.shape[0]

    def __str__(self):
        return "Trace Fidelity"
