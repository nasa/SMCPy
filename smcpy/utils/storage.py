import numpy as np

from abc import abstractmethod

class BaseStorage:

    def __init__(self):
        self.is_restart = False

    @abstractmethod
    def get_unnormalized_log_weights(self):
        '''
        Gets unnormalized log weights for all particles in all steps. Returns
        a list of with length = # steps and each element a (# particles, 1)
        2D array.
        '''

    def estimate_marginal_log_likelihoods(self):
        sum_un_log_wts = [np.array([0])]
        sum_un_log_wts += [self._logsum(p.unnorm_log_weights) for p in self]
        return np.cumsum(sum_un_log_wts)

    @staticmethod
    def _logsum(Z):
        Z = -np.sort(-Z, axis=0) # descending
        Z0 = Z[0, :]
        Z_shifted = Z[1:, :] - Z0
        return Z0 + np.log(1 + np.sum(np.exp(Z_shifted), axis=0))


class InMemoryStorage(BaseStorage):

    def __init__(self):
        super().__init__()
        self._step_list = []

    def __getitem__(self, idx):
        return self._step_list[idx]

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < len(self._step_list):
            output = self._step_list[self._idx]
            self._idx += 1
            return output
        else:
            raise StopIteration

    def save_step(self, step):
        self._step_list.append(step)
