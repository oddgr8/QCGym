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
            params : iterable of shape (timesteps, ...)
                Array of one control parameter for each timestep

        Returns
        -------
            arr : ndarray of shape(timesteps*mesh, ...)
                Array of interpolated control parameters
        """

        return np.repeat(np.array(params), self.mesh, axis=0)

    def __str__(self):
        return f"IdentityInterp(mesh={self.mesh})"
