import abc
import numpy as np
import warnings

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta("ABC", (object,), {"__slots__": ()})


class PathBase:
    def __init__(self, proposal):
        self._phi_list = [0]
        self._proposal = proposal

    @property
    def phi(self):
        return self._phi_list[-1]

    @phi.setter
    def phi(self, phi):
        if phi <= self._phi_list[-1]:
            raise ValueError(
                "phi updates must be monotonic; " f"tried {self.phi} -> {phi}"
            )
        self._phi_list.append(phi)

    @property
    def previous_phi(self):
        try:
            return self._phi_list[-2]
        except IndexError:
            return None

    @property
    def delta_phi(self):
        try:
            return self._phi_list[-1] - self._phi_list[-2]
        except IndexError:
            return None

    def undo_phi_set(self):
        self._phi_list = self._phi_list[:-1]

    @property
    def proposal(self):
        return self._proposal

    @abc.abstractmethod
    def logpdf(self, inputs, log_like, log_prior):
        return None

    @abc.abstractmethod
    def inc_log_weights(self, inputs, log_like, log_prior, delta_phi):
        return None

    @staticmethod
    def _log_prob_sum(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = x.sum(axis=1, keepdims=True)
            y[np.isnan(y)] = -np.inf  # probability 0/0 => 0
        return y


class GeometricPath(PathBase):
    def __init__(self, proposal=None, required_phi=1):
        super().__init__(proposal)
        self._lambda = None
        self.required_phi_list = required_phi

    @property
    def required_phi_list(self):
        return self._required_phi_list.copy()

    @required_phi_list.setter
    def required_phi_list(self, phi):
        if isinstance(phi, float) or isinstance(phi, int):
            phi = [phi]
        self._required_phi_list = sorted([p for p in phi if p < 1])
        self._lambda = min(self._required_phi_list + [1])

    def logpdf(self, inputs, log_like, log_prior):
        log_p = self._get_proposal_logpdf(inputs, log_prior)
        args = log_like, log_prior, log_p
        return self._log_prob_sum(self._eval_target(*args, self.phi))

    def inc_log_weights(self, inputs, log_like, log_prior):
        log_p = self._get_proposal_logpdf(inputs, log_prior)
        args = log_like, log_prior, log_p
        numer = self._eval_target(*args, self.phi)
        denom = self._eval_target(*args, self.previous_phi)
        return self._log_prob_sum(np.hstack((numer, -denom)))

    def _eval_target(self, log_like, log_prior, log_p, phi):
        prior_exp = min(1.0, phi / self._lambda)
        prop_exp = max(0.0, (self._lambda - phi) / self._lambda)

        target = np.hstack(
            (
                log_like * phi,
                log_prior * prior_exp if prior_exp > 0 else np.zeros_like(log_p),
                log_p * prop_exp if prop_exp > 0 else np.zeros_like(log_p),
            )
        )

        return target

    def _get_proposal_logpdf(self, inputs, log_prior):
        return (
            self._proposal.logpdf(inputs).reshape(-1, 1)
            if self._proposal
            else log_prior
        )

# class MultiFidelityPath(PathBase):
#     """
#     Defines a multi-fidelity annealing path for Sequential Monte Carlo (SMC).
    
#     This class constructs a path that transitions from a proposal distribution 
#     (or prior) to the true target distribution. It uses a bridging parameter `phi` 
#     and a threshold `_lambda` to control the exponents applied to the likelihood, 
#     prior, and proposal distributions at each step.

#     Attributes:
#         _lambda (float): The threshold bridging parameter, determined as the minimum 
#             value from `required_phi_list` (or 1 if none are provided). It dictates 
#             when the transition shifts fully from the proposal to the prior/likelihood.
#         _required_phi_list (list): A sorted list of required intermediate `phi` levels.
#     """

#     def __init__(self, proposal=None, required_phi=1):
#         """
#         Initializes the MultiFidelityPath.

#         Args:
#             proposal (object, optional): An object representing the proposal 
#                 distribution. It must implement a `logpdf` method. Defaults to None.
#             required_phi (float, int, or list, optional): A specific `phi` value or 
#                 a list of `phi` values that the SMC schedule must hit. Defaults to 1.
#         """
#         super().__init__(proposal)
#         self._lambda = None
#         # This will utilize the setter method to initialize the list and lambda
#         self.required_phi_list = required_phi

#     @property
#     def required_phi_list(self):
#         """
#         list: A copy of the required `phi` levels strictly less than 1.
#         """
#         return self._required_phi_list.copy()

#     @required_phi_list.setter
#     def required_phi_list(self, phi):
#         """
#         Sets the required `phi` list and recalculates the `_lambda` threshold.

#         Args:
#             phi (float, int, or list): The required phi levels to be set.
#         """
#         if isinstance(phi, float) or isinstance(phi, int):
#             phi = [phi]
#         # Only keep phi values strictly less than 1, sorted in ascending order
#         self._required_phi_list = sorted([p for p in phi if p < 1])
#         # _lambda is the smallest required phi, or 1.0 if the list is empty
#         self._lambda = min(self._required_phi_list + [1])

#     def logpdf(self, inputs, log_like, log_prior):
#         """
#         Calculates the log probability density of the target at the current `phi`.

#         Args:
#             inputs (ndarray): The current particle states/inputs.
#             log_like (ndarray): The log-likelihood of the particles.
#             log_prior (ndarray): The log-prior density of the particles.

#         Returns:
#             ndarray: The log-target probability for each particle at `self.phi`.
#         """
#         log_p = self._get_proposal_logpdf(inputs, log_prior)
#         args = log_like, log_prior, log_p
        
#         # Evaluates the target components and sums them (assuming _log_prob_sum exists in PathBase)
#         return self._log_prob_sum(self._eval_target(*args, self.phi))

#     def inc_log_weights(self, inputs, log_like, log_prior):
#         """
#         Calculates the incremental log weights to transition from `previous_phi` to `phi`.

#         This is used during the SMC reweighting step to update particle weights as 
#         the target distribution evolves.

#         Args:
#             inputs (ndarray): The current particle states/inputs.
#             log_like (ndarray): The log-likelihood of the particles.
#             log_prior (ndarray): The log-prior density of the particles.

#         Returns:
#             ndarray: The incremental log weights for the particles.
#         """
#         log_p = self._get_proposal_logpdf(inputs, log_prior)
#         args = log_like, log_prior, log_p
        
#         # Calculate the log target for the current step
#         numer = self._eval_target(*args, self.phi)
#         # Calculate the log target for the previous step
#         denom = self._eval_target(*args, self.previous_phi)
        
#         # Weight update is target(phi) / target(previous_phi) 
#         # In log space, this is numer - denom. 
#         # hstack is used to format the arrays before applying _log_prob_sum.
#         return self._log_prob_sum(np.hstack((numer, -denom)))

#     def _eval_target(self, log_like, log_prior, log_p, phi):
#         """
#         Evaluates the components of the intermediate target distribution for a given `phi`.

#         The target is constructed using a mixture of the proposal, prior, and likelihood 
#         governed by `phi` and `_lambda`. 
#         - `prior_exp`: Scales the prior contribution up as `phi` approaches `_lambda`.
#         - `prop_exp`: Scales the proposal contribution down as `phi` approaches `_lambda`.
#         - Likelihood is scaled directly by `phi`.

#         Args:
#             log_like (ndarray): The log-likelihood array.
#             log_prior (ndarray): The log-prior array.
#             log_p (ndarray): The log-proposal array.
#             phi (float): The current bridging parameter/temperature.

#         Returns:
#             ndarray: A concatenated array of the scaled log-likelihood, scaled log-prior, 
#                 and scaled log-proposal.
#         """
#         # Prior exponent linearly increases to 1.0 when phi reaches _lambda
#         prior_exp = min(1.0, phi / self._lambda)
        
#         # Proposal exponent linearly decreases to 0.0 when phi reaches _lambda
#         prop_exp = max(0.0, (self._lambda - phi) / self._lambda)

#         target = np.hstack(
#             (
#                 log_like * phi,
#                 log_prior * prior_exp if prior_exp > 0 else np.zeros_like(log_p),
#                 log_p * prop_exp if prop_exp > 0 else np.zeros_like(log_p),
#             )
#         )

#         return target

#     def _get_proposal_logpdf(self, inputs, log_prior):
#         """
#         Retrieves the log probability density of the proposal distribution.

#         If no proposal distribution was provided during initialization, it defaults 
#         to using the log-prior.

#         Args:
#             inputs (ndarray): The particle states/inputs to evaluate.
#             log_prior (ndarray): The log-prior, used as a fallback.

#         Returns:
#             ndarray: The log-probability density of the proposal.
#         """
#         return (
#             self._proposal.logpdf(inputs).reshape(-1, 1)
#             if self._proposal
#             else log_prior
#         )