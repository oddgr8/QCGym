import numpy as np
import logging
logger = logging.getLogger(__name__)


class IdentityInterpolator(object):
    """
    Piecewise constant interpolator.

    Parameters
    ----------
        mesh : int
            How many subintervals to divide each timestep into
    """

    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, params):
        """
        Return interpolated array.

        Parameters
        ----------
            params : ndarray of shape(x,)
                Array of one control parameter for each timestep

        Returns
        -------
            arr : ndarray of shape(x*mesh,)
                Array of interpolated control parameters
        """
        return np.repeat(params, mesh, axis=0)
