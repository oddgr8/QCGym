from QCGym.fidelities.generic_fidelity import GenericFidelity
import numpy as np


class TraceFidelity(GenericFidelity):
    """
    Trace fidelity among unitaries

    F = trace(unitaries, target)/comp_dim
    """

    def __call__(self, unitary, target):
        return np.abs(np.matmul(unitary.T, target))/unitary.shape[0]
