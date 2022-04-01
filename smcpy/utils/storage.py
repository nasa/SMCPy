import h5py
import numpy as np
import os

from abc import abstractmethod

from ..smc.particles import Particles
from .context_manager import ContextManager


class BaseStorage(ContextManager):

    def __init__(self):
        self.is_restart = False

    def estimate_marginal_log_likelihoods(self):
        sum_un_log_wts = [0]
        sum_un_log_wts += [p.total_unnorm_log_weight for p in self]
        return np.cumsum(sum_un_log_wts)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def __iter__(self):
        self._idx = 0
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def save_step(self, step):
        raise NotImplementedError



class InMemoryStorage(BaseStorage):

    def __init__(self):
        super().__init__()
        self._step_list = []
        self._phi_sequence = []

    @property
    def phi_sequence(self):
        return self._phi_sequence

    def __getitem__(self, idx):
        return self._step_list[idx]

    def __next__(self):
        if self._idx < len(self._step_list):
            output = self._step_list[self._idx]
            self._idx += 1
            return output
        else:
            raise StopIteration

    def __len__(self):
        return len(self._step_list)

    def save_step(self, step):
        self._step_list.append(step)
        self._phi_sequence.append(step.attrs['phi'])



class HDF5Storage(BaseStorage):

    def __init__(self, filename, mode='a'):
        super().__init__()
        self._filename = filename
        self._mode = mode
        self._len = 0
        if os.path.exists(filename):
            if mode != 'w':
                self._init_length_on_restart()
                self.is_restart = True
            else:
                os.remove(filename)

    @property
    def phi_sequence(self):
        h5 = self._open_h5('r')
        phi_sequence = [h5[i].attrs['phi'] for i in sorted(h5.keys(), key=int)]
        self._close(h5)
        return phi_sequence

    def save_step(self, step):
        h5 = self._open_h5('a')

        step_grp = h5.create_group(str(len(self)))
        step_grp.attrs['phi'] = step.attrs['phi']
        step_grp.attrs['total_unnorm_log_weight'] = step.total_unnorm_log_weight
        step_grp.create_dataset('log_likes', data=step.log_likes)
        step_grp.create_dataset('log_weights', data=step.log_weights)

        param_grp = step_grp.create_group('params')
        for key, array in step.param_dict.items():
            param_grp.create_dataset(key, data=array)

        self._close(h5)

    def _open_h5(self, mode=None):
        if not mode:
            mode = self._mode
        h5 = h5py.File(self._filename, mode)
        self._len = len(h5.keys())
        return h5

    def _close(self, h5):
        self._len = len(h5.keys())
        h5.close()

    def __getitem__(self, idx):
        h5 = self._open_h5('r')
        step_grp = h5[self._format_index(idx)]

        total_unlw = step_grp.attrs['total_unnorm_log_weight']
        kwargs = {k: v[:] for k, v in step_grp.items() if k != 'params'}
        kwargs['params'] = {k: v[:] for k, v in step_grp['params'].items()}
        particles = Particles(**kwargs)
        particles._total_unlw = total_unlw

        self._close(h5)
        return particles

    def _format_index(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f'index {idx} out of range.')
        return str(idx)

    def __next__(self):
        try:
            output = self[self._idx]
            self._idx += 1
            return output
        except IndexError:
            raise StopIteration

    def __len__(self):
        return self._len

    def _init_length_on_restart(self):
        h5 = self._open_h5('r')
        self._close(h5)
