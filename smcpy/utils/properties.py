from checks import Checks


class Properties(Checks):


    def __init__(self):
        super(Properties, self).__init__()
        self.num_particles = 1
        self.num_time_steps = 1
        self.temp_schedule = [0.0, 1.0]
        self.num_mcmc_steps = 1
        self.ess_threshold = 0
        self.autosaver = None
        self.restart_time_step = 0


    @property
    def num_particles(self):
        return self._num_particles


    @num_particles.setter
    def num_particles(self, num_particles):
        input_ = 'num_particles'
        if not self._is_integer(input_):
            self._raise_type_error(input_, 'integer')
        if self._is_negative(input_):
            self._raise_negative_error(input_)
        self._num_particles = num_particles
        return None


    @property
    def num_time_steps(self):
        return self._num_time_steps


    @num_time_steps.setter
    def num_time_steps(self, num_time_steps):
        input_ = 'num_time_steps'
        if not self._is_integer(input_):
            self._raise_type_error(input_, 'integer')
        if self._is_negative(input_):
            self._raise_negative_error(input_)
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
        input_ = 'num_mcmc_steps'
        if not self._is_integer(input_):
            self._raise_type_error(input_, 'integer')
        if self._is_negative(input_):
            self._raise_negative_error(input_)
        self._num_mcmc_steps = num_mcmc_steps
        return None


    @property
    def ess_threshold(self):
        return self._ess_threshold


    @ess_threshold.setter
    def ess_threshold(self, ess_threshold):
        if ess_threshold is None:
            ess_threshold = 0.5 * self.num_particles
        if not self._is_integer_or_float(ess_threshold):
            self.raise_type_error(input_, 'float or int')
        self._ess_threshold = ess_threshold
        return None


    @property
    def autosaver(self):
        return self._autosaver


    @autosaver.setter
    def autosaver(self, autosave_file):
        import pdb; pdb.set_trace()
        if not self._is_string_or_none(autosave_file):
            self._raise_type_error('autosave_file', 'string or None')
        if self._rank == 0:
            self._autosaver = HDF5Storage(autosave_file, mode='w')
        else:
            self._autosaver = None
        return None


    @property
    def restart_time_step(self):
        return self._restart_time_step


    @restart_time_step.setter
    def restart_time_step(self, restart_time_step):
        input_ = 'restart_time_step'
        if not self._is_integer(input_):
            self._raise_type_error(input_, 'integer')
        if self._is_negative(input_):
            self._raise_negative_error(input_)
        if restart_time_step > self.num_time_steps:
            raise ValueError('restart_time_step cannot be > num_time_steps')
        self._restart_time_step = restart_time_step
        return None
