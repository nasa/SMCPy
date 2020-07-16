import inspect
import numpy as np
import pymc3

from copy import copy

from .kernel_base import MCMCKernel
from .pymc3_step_methods import SMCStepMethod

import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)


class PyMC3Kernel(MCMCKernel):

    def __init__(self, pymc3_model, step_method):
        self._pymc3_model = pymc3_model
        self.step_method = step_method

    @property
    def pymc3_model(self):
        return copy(self._pymc3_model)

    @property
    def step_method(self):
        return self._step_method

    @step_method.setter
    def step_method(self, step_method):
        is_class = inspect.isclass(step_method)
        is_smc_step = SMCStepMethod in step_method.__bases__
        if not is_class or not is_smc_step:
            raise TypeError
        with self.pymc3_model:
            self._step_method = step_method()

    def get_data(self):
        return self._pymc3_model.observed_RVs[0].observations

    def get_log_likelihood(self, params):
        return float(self._pymc3_model.observed_RVs[0].logp(params))

    def sample(self, num_samples, init_params, cov, phi):
        with self.pymc3_model:
            for i, method in enumerate(self.step_method.methods):
                method.phi = phi
                method.S = np.sqrt(cov[i, i]).reshape(1,)
            self._last_trace = pymc3.sampling.sample(draws=num_samples,
                                      step=self.step_method, chains=1,
                                      cores=1, start=init_params, tune=False,
                                      discard_tuned_samples=False,
                                      progressbar=False)

    def get_all_trace_values(self):
        param_names = self._last_trace.varnames
        param_names = [pn for pn in param_names if not pn.endswith('__')]
        return {pn: list(self._last_trace.get_values(pn)) for pn in param_names}

    def get_final_trace_values(self):
        trace = self.get_all_trace_values()
        return {pn: values[-1] for pn, values in trace.items()}

    def sample_from_prior(self, size=1):
        random_sample = {}
        for param in self.pymc3_model.vars:
            param_name = param.name
            if not param_name.endswith('__'):
                random_sample[param_name] = param.random(size=size)
        return random_sample