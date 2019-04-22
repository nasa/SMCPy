from ..utils.single_rank_comm import SingleRankComm


class ParticleUpdater():

    def __init__(self, step, ess_threshold, mpi_comm=SingleRankComm()):
        self.step = step
        self.ess_threshold = ess_threshold
        self._comm = mpi_comm
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    def update_particles(self, temperature_step):
        if self._rank == 0:
            self.step.normalize_step_log_weights()
            self._update_log_weights(temperature_step)
            self._resample_if_needed()
        else:
            self.step = None
        return self.step

    def _update_log_weights(self, temperature_step):
        for p in self.step.get_particles():
            print p.log_weight, p.log_like, temperature_step
            temp_weight = p.log_weight
            p.log_weight = temp_weight + p.log_like * temperature_step
        return None

    def _resample_if_needed(self):  # issue here
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
