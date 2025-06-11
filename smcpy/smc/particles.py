"""
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRessED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNess FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLess THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
"""

import numpy as np
import copy
import functools
from smcpy.utils.checks import Checks


def package_for_user(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        Packages array output in a dictionary with keys = parameter names and
        values = array columns UNLESS kwarg package=False is passed to the
        decorated function.
        """
        if "package" in kwargs.keys() and kwargs["package"] == False:
            return func(self)

        else:
            outputs = func(self, *args, **kwargs)
            names = self.param_names
            return {name: output for name, output in zip(names, outputs)}

    return wrapper


class Particles(Checks):
    """
    A container for particles during sequential monte carlo (SMC) sampling.
    """

    def __init__(self, params, log_likes, log_weights):
        """
        :param params: model parameters; keys are parameter names and values
            are particle values
        :type params: dictionary of 1D arrays or lists
        :param log_likes: natural log of particle likelihoods
        :type log_likes: 1D array or list
        :param log_weights: natural log of particle weights; weights will be
            automatically normalized when set to avoid potential misuse
        :type log_weights: 1D array or list
        """
        self._param_names = None
        self._num_particles = None

        self._set_params(params)
        self._set_log_likes(log_likes)
        self._set_and_norm_log_weights(log_weights)

        self.attrs = {"total_unnorm_log_weight": self._logsum(log_weights)}

    @property
    def params(self):
        return self._params

    def _set_params(self, params):
        if not self._is_dict(params):
            raise TypeError('"params" must be dict of array-like objects.')
        self._param_names = tuple(params.keys())
        self._params = np.vstack(list(params.values())).T
        self._num_particles = self._params.shape[0]

    @property
    def param_dict(self):
        return dict(zip(self._param_names, self._params.T))

    @property
    def log_likes(self):
        return self._log_likes

    def _set_log_likes(self, log_likes):
        log_likes = np.array(log_likes).reshape(-1, 1)
        if log_likes.shape[0] != self._num_particles:
            raise ValueError(
                "log_likes.shape[0] != number particles: "
                f"{log_likes.shape[0]} != {self._num_particles}"
            )
        self._log_likes = log_likes

    @property
    def total_unnorm_log_weight(self):
        return self.attrs["total_unnorm_log_weight"]

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def weights(self):
        return self._weights

    def _set_and_norm_log_weights(self, log_weights):
        log_weights = np.array(log_weights).reshape(-1, 1)
        if log_weights.shape[0] != self._num_particles:
            raise ValueError(
                "log_weights.shape[0] != number particles: "
                f"{log_weights.shape[0]} != {self._num_particles}"
            )
        self._log_weights = self._normalize_log_weights(log_weights)
        self._weights = np.exp(self._log_weights)

    @property
    def param_names(self):
        return self._param_names

    @property
    def num_particles(self):
        return self._num_particles

    def _normalize_log_weights(self, log_weights):
        """
        Normalizes log weights, and then transforms back into log space
        """
        shifted_weights = np.exp(log_weights - np.max(log_weights))
        normalized_weights = shifted_weights / np.sum(shifted_weights)
        log_zero_weights = np.ones(normalized_weights.shape) * -np.inf
        return np.log(
            normalized_weights, out=log_zero_weights, where=normalized_weights > 0
        )

    def copy(self):
        """
        Returns a copy of the entire step class.
        """
        return copy.deepcopy(self)

    def compute_ess(self):
        """
        Computes the effective sample size (ess) of the step based on log weight
        """
        return 1 / np.sum(self.weights**2)

    @package_for_user
    def compute_mean(self):
        """
        Returns the estimated mean of each parameter.
        """
        return np.sum(self.params * self.weights, axis=0)

    @package_for_user
    def compute_variance(self):
        """
        Returns the estimated variance of each parameter. Uses weighted
        sample formula https://en.wikipedia.org/wiki/Sample_mean_and_covariance
        """
        means = self.compute_mean(package=False)
        norm = 1 - np.sum(self.weights**2)
        return np.sum(self.weights * (self.params - means) ** 2, axis=0) / norm

    @package_for_user
    def compute_std_dev(self):
        """
        Returns the estimated standard deviation of each parameter.
        """
        var = self.compute_variance(package=False)
        return np.sqrt(var)

    def compute_covariance(self):
        """
        Estimates the covariance matrix.
        """
        cov = np.cov(self.params.T, ddof=0, aweights=self.weights.flatten())

        if cov.shape == ():
            cov = cov.reshape(1, 1)

        return cov

    @staticmethod
    def _logsum(Z):
        Z = -np.sort(-np.array(Z), axis=0).flatten()  # descending
        Z0 = Z[0]
        Z_shifted = Z[1:] - Z0
        return Z0 + np.log(1 + np.sum(np.exp(Z_shifted), axis=0))
