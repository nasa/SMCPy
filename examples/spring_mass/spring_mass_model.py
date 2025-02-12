import numpy as np
from scipy.integrate import odeint

# ------------------------------------------------------
# Helper function to use scipy integrator in model class


def mass_spring(state, t, K, g):
    """
    Return velocity/acceleration given velocity/position and values for
    stiffness and mass
    """

    # unpack the state vector
    x = state[0]
    xd = state[1]

    # compute acceleration xdd
    xdd = -K * x + g

    # return the two state derivatives
    return [xd, xdd]


# ------------------------------------------------------


class SpringMassModel:
    """
    Defines Spring Mass model with 2 free params (spring stiffness, k & mass, m)
    """

    def __init__(self, state0=None, time_grid=None):
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_grid is None:
            time_grid = np.arange(0.0, 10.0, 0.1)

        self._state0 = state0
        self._t = time_grid

    def evaluate(self, params):
        """
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid
        """
        results = []
        for p in params:
            K = p[0]
            g = p[1]
            output = odeint(mass_spring, self._state0, self._t, args=(K, g))
            results.append(output[:, 0])
        return np.array(results)
