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
