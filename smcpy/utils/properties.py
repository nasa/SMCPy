from checks import Checks
from ..hdf5.hdf5_storage import HDF5Storage
from ..smc.smc_step import SMCStep


class Properties(Checks):
    def __init__(self):
        super(Properties, self).__init__()
        self._num_particles = 1
        self._num_time_steps = 1
        self._temp_schedule = [0.0, 1.0]
        self._num_mcmc_steps = 1
        self._ess_threshold = 0
        self._autosaver = None
        self._restart_time_step = 0
        self._particle_chain = SMCStep()

    @property
    def num_particles(self):
        return self._num_particles

    @num_particles.setter
    def num_particles(self, num_particles):
        input_ = "num_particles"
        if not self._is_integer(num_particles):
            self._raise_type_error(input_, "integer")
        if self._is_negative(num_particles):
            self._raise_negative_error(input_)
        if self._is_zero(num_particles):
            self._raise_zero_error(input_)
        self._num_particles = num_particles
        return None

    @property
    def num_time_steps(self):
        return self._num_time_steps

    @num_time_steps.setter
    def num_time_steps(self, num_time_steps):
        input_ = "num_time_steps"
        if not self._is_integer(num_time_steps):
            self._raise_type_error(input_, "integer")
        if self._is_negative(num_time_steps):
            self._raise_negative_error(input_)
        if self._is_zero(num_time_steps):
            self._raise_zero_error(input_)
        self._num_time_steps = num_time_steps
        return None

    @property
    def temp_schedule(self):
        return self._temp_schedule

    @temp_schedule.setter
    def temp_schedule(self, temp_schedule):
        self._temp_schedule = temp_schedule

    @property
    def num_mcmc_steps(self):
        return self._num_mcmc_steps

    @num_mcmc_steps.setter
    def num_mcmc_steps(self, num_mcmc_steps):
        input_ = "num_mcmc_steps"
        if not self._is_integer(num_mcmc_steps):
            self._raise_type_error(input_, "integer")
        if self._is_negative(num_mcmc_steps):
            self._raise_negative_error(input_)
        if self._is_zero(num_mcmc_steps):
            self._raise_zero_error(input_)
        self._num_mcmc_steps = num_mcmc_steps
        return None

    @property
    def ess_threshold(self):
        return self._ess_threshold

    @ess_threshold.setter
    def ess_threshold(self, ess_threshold):
        input_ = "ess_threshold"
        if ess_threshold is None:
            ess_threshold = 0.5 * self.num_particles
        if not self._is_integer_or_float(ess_threshold):
            self._raise_type_error(input_, "float or int")
        if self._is_negative(ess_threshold):
            self._raise_negative_error(input_)
        self._ess_threshold = ess_threshold
        return None

    @property
    def autosaver(self):
        return self._autosaver

    @autosaver.setter
    def autosaver(self, autosave_file):
        if not self._is_string_or_none(autosave_file):
            self._raise_type_error("autosave_file", "string or None")
        if self._rank == 0 and autosave_file is not None:
            self._autosaver = HDF5Storage(autosave_file, mode="w")
        else:
            self._autosaver = None
        return None

    @property
    def restart_time_step(self):
        return self._restart_time_step

    @restart_time_step.setter
    def restart_time_step(self, restart_time_step):
        input_ = "restart_time_step"
        if not self._is_integer(restart_time_step):
            self._raise_type_error(input_, "integer")
        if self._is_negative(restart_time_step):
            self._raise_negative_error(input_)
        if restart_time_step > self.num_time_steps:
            self._raise_out_of_bounds_error(input_)
        self._restart_time_step = restart_time_step
        return None

    @property
    def particle_chain(self):
        return self._particle_chain

    @particle_chain.setter
    def particle_chain(self, particle_chain):
        input_ = "particle_chain"
        if not isinstance(particle_chain, SMCStep) and not self._is_none(
            particle_chain
        ):
            self._raise_type_error(input_, "SMCStep instance or None")
        self._particle_chain = particle_chain
        return None
