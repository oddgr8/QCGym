from QCGym.interpolators.identity import IdentityInterpolator


class GenericHamiltonian(object):
    """
    Hamiltonian base class.

    Parameters
    ----------
        smoothing : Interpolator
            How to interpolate parameters inside timesteps
    """
    action_space = None

    def __init__(self, smoothing=IdentityInterpolator(10)):
        self.smoothing = smoothing

    def __call__(self, control_params):
        '''
        Returns Hamiltonian over time

                Parameters:
                        control_params : iterable of shape(timesteps, ...shape of action_space)

                Returns:
                        hamiltonians : ndarray of shape(time_steps*mesh, ...shape_of_hamiltonian)
        '''
        raise NotImplementedError(
            "This is a hamiltonian base class. Use other specialized hamiltonians.")

    def __str__(self):
        return f"Gen Hamilotonian-{self.smoothing}"
