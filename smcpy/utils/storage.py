import h5py
import numpy as np
import os

from abc import abstractmethod

from smcpy.smc.particles import Particles


class BaseStorage:

    def __init__(self):
        self.is_restart = False

    def estimate_marginal_log_likelihoods(self):
        sum_un_log_wts = [0]
        sum_un_log_wts += [p.total_unnorm_log_weight for p in self]
        return np.cumsum(sum_un_log_wts)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def __iter__(self):
        self._idx = 0
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def save_step(self, step):
        pass



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

    def save_step(self, step):
        self._step_list.append(step)
        self._phi_sequence.append(step.attrs['phi'])



class HDF5Storage(BaseStorage):

    def __init__(self, filename, mode='a'):
        super().__init__()
        self._filename = filename
        self._mode = mode
        if os.path.exists(filename) and mode != 'w':
            self.is_restart = True

    @property
    def phi_sequence(self):
        h5 = self._open_h5()
        phi_sequence = [h5[i].attrs['phi'] for i in sorted(h5.keys())]
        h5.close()
        return phi_sequence

    def save_step(self, step):
        h5 = self._open_h5()
        last_index = int(max([i for i in h5.keys()] + ['-1']))
        step_grp = h5.create_group(str(last_index + 1))
        step_grp.attrs['phi'] = step.attrs['phi']
        step_grp.attrs['total_unnorm_log_weight'] = step.total_unnorm_log_weight
        step_grp.create_dataset('params', data=step.params)
        step_grp.create_dataset('log_likes', data=step.log_likes)
        step_grp.create_dataset('log_weights', data=step.log_weights)
        h5.close()

    def _load_existing_phi_sequence(self):
        h5 = self._open_h5()

    def _open_h5(self):
        return h5py.File(self._filename, self._mode)

    def __getitem__(self, idx):
        h5 = self._open_h5()
        step_grp = h5[str(idx)]

        total_unlw = step_grp.attrs['total_unnorm_log_weight']
        kwargs = {key: val[:] for key, val in step_grp.items()}
        particles = Particles(**kwargs)
        particles.total_unnorm_log_weight = total_unlw

        h5.close()
        return particles

    def __next__(self):
        try:
            output = self[self._idx]
            self._idx += 1
            return output
        except KeyError:
            raise StopIteration


