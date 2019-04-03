from copy import copy
from pymc import Normal
from ..particles.particle import Particle
import numpy as np


class ParticleInitializer():

    def __init__(self, mcmc):
        self._mcmc = mcmc

    def initialize_particles(self, measurement_std_dev):
        m_std = measurement_std_dev
        self._mcmc.generate_pymc_model(fix_var=True, std_dev0=m_std)
        num_particles_per_partition = self._get_num_particles_per_partition()
        particles = []
        prior_variables = self._create_prior_random_variables()
        if self.proposal_center is not None:
            proposal_variables = self._create_proposal_random_variables()
        else:
            proposal_variables = None
        for _ in range(num_particles_per_partition):
            part = self._create_particle(prior_variables, proposal_variables)
            particles.append(part)
        return particles

    def _get_num_particles_per_partition(self,):
        num_particles_per_partition = self.num_particles / self._size
        remainder = self.num_particles % self._size
        overtime_ranks = range(remainder)
        if self._rank in overtime_ranks:
            num_particles_per_partition += 1
        return num_particles_per_partition

    def _create_prior_random_variables(self,):
        mcmc = copy(self._mcmc)
        random_variables = dict()
        for key in mcmc.params.keys():
            index = mcmc.pymc_mod_order.index(key)
            random_variables[key] = mcmc.pymc_mod[index]
        return random_variables

    def _create_proposal_random_variables(self,):
        centers = self.proposal_center
        scales = self.proposal_scales
        random_variables = dict()
        for key in self._mcmc.params.keys():
            variance = (centers[key] * scales[key])**2
            random_variables[key] = Normal(key, centers[key], 1 / variance)
        return random_variables

    def _create_particle(self, prior_variables, prop_variables=None):
        if prop_variables is None:
            params = self._sample_random_variables(prior_variables)
            prior_logp = self._compute_log_prob(prior_variables)
            prop_logp = prior_logp
        else:
            params = self._sample_random_variables(prop_variables)
            prop_logp = self._compute_log_prob(prop_variables)
            self._set_random_variables_value(prior_variables, params)
            prior_logp = self._compute_log_prob(prior_variables)
        log_like = self._evaluate_likelihood(params)
        temp_step = self.temp_schedule[self._start_time_step]
        log_weight = log_like * temp_step + prior_logp - prop_logp
        return Particle(params, np.exp(log_weight), log_like)

    def _sample_random_variables(self, random_variables):
        param_keys = self._mcmc.params.keys()
        params = {key: random_variables[key].random() for key in param_keys}
        return params

    @staticmethod
    def _set_random_variables_value(random_variables, params):
        for key, value in params.iteritems():
            random_variables[key].value = value
        return None

    @staticmethod
    def _compute_log_prob(random_variables):
        param_log_prob = np.sum([rv.logp for rv in random_variables.values()])
        return param_log_prob

    def _evaluate_likelihood(self, param_vals):
        '''
        Note: this method performs 1 model evaluation per call.
        '''
        mcmc = copy(self._mcmc)
        for key, value in param_vals.iteritems():
            index = mcmc.pymc_mod_order.index(key)
            mcmc.pymc_mod[index].value = value
        results_index = mcmc.pymc_mod_order.index('results')
        results_rv = mcmc.pymc_mod[results_index]
        log_like = results_rv.logp
        return log_like
