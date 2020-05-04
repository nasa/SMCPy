import inspect
import pymc3

from copy import copy

from .translator_base import Translator
from .pymc3_step_methods import SMCStepMethod

class PyMC3Translator(Translator):

    def __init__(self, pymc3_model, step_method):
        self._pymc3_model = pymc3_model
        self._check_step_method(step_method)
        self._step_method = step_method

    @property
    def pymc3_model(self):
        return copy(self._pymc3_model)

    def get_data(self):
        return self._pymc3_model.observed_RVs[0].observations

    def get_log_likelihood(self, params):
        return float(self._pymc3_model.observed_RVs[0].logp(params))

    def sample(self, samples, init_params, phi):
        with self.pymc3_model:
            self._last_step_method = self._step_method(phi=phi)
            self._last_trace = pymc3.sampling.sample(draws=samples,
                                      step=self._last_step_method, chains=1,
                                      cores=1, start=init_params, tune=0,
                                      discard_tuned_samples=False)

    def get_final_trace_values(self):
        param_names = self._last_trace.varnames
        param_names = [pn for pn in param_names if not pn.endswith('__')]
        return {pn: self._last_trace.get_values(pn)[-1] for pn in param_names}


    def sample_from_prior(self, size=1):
        random_sample = {}
        for param in self.pymc3_model.vars:
            param_name = param.name
            if not param_name.endswith('__'):
                random_sample[param_name] = param.random(size=size)
        return random_sample


    @staticmethod
    def _check_step_method(step_method):
        is_class = inspect.isclass(step_method)
        is_smc_step = SMCStepMethod in step_method.__bases__
        if not is_class or not is_smc_step:
            raise TypeError
