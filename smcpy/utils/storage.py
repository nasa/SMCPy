import numpy as np

from abc import abstractmethod

class BaseStorage:

    def __init__(self):
        self.is_restart = False

    def estimate_marginal_log_likelihoods(self):
        sum_un_log_wts = [0]
        sum_un_log_wts += [(p.compute_total_unnorm_log_wt()) for p in self]
        return np.cumsum(sum_un_log_wts)


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
