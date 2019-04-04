import numpy as np


class ParticleUpdater():

    def __init__(self, step, mpi_comm):
        self.step = step
        self._comm = mpi_comm
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    def update_particles(self, temperature_step):
        if self._rank == 0:
            # self._initialize_new_particles()
            self.update_weights(temperature_step)
            self.step.normalize_step_weights()
            self.resample_if_needed()
            new_particles = self._partition_new_particles()
        else:
            new_particles = [None]
        new_particles = self._comm.scatter(new_particles, root=0)
        return list(new_particles)

    def update_weights(self, temperature_step):
        for p in self.step.get_particles():
            p.weight = np.exp(np.log(p.weight) + p.log_like * temperature_step)
        return None

    def resample_if_needed(self):
        '''
        Checks if ess below threshold; if yes, resample with replacement.
        '''
        self._ess = self.step.compute_ess()
        if self._ess < self.ess_threshold:
            self._resample_status = "Resampling..."
            self.step.resample()
        else:
            self._resample_status = "No resampling"
        return None

    def _partition_new_particles(self):
        partitions = np.array_split(self.step.get_particles(),
                                    self._size)
        return partitions

    def _initialize_new_particles(self):
        new_particles = self.step.copy_step()
        self.step.set_particles(new_particles)
        return None
