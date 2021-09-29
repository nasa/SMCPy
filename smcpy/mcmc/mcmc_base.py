import cupy
import numpy as np
import nvtx

from abc import ABC, abstractmethod
from tqdm import tqdm

import smcpy.utils.global_imports as gi


class MCMCBase(ABC):

    def __init__(self, model, data, log_like_args, log_like_func):
        '''
        :param model: maps inputs to outputs
        :type model: callable
        :param data: data corresponding to model outputs
        :type data: 1D array
        :param log_like_args: any fixed parameters that define the likelihood
            function (e.g., standard deviation for a Gaussian likelihood).
        :type log_like_args: 1D array or None
        :param log_like_func: log likelihood function that takes inputs, model,
            data, and hyperparameters and returns log likelihoods
        :type log_like_func: callable
        '''
        self._eval_model = model
        self._data = data
        self._log_like_func = log_like_func(self.evaluate_model, data,
                                            log_like_args)

    def smc_metropolis(self, inputs, num_samples, cov, phi):
        with nvtx.annotate(message='convert inputs cov', color='turquoise'):
            inputs = gi.num_lib.asarray(inputs)
            cov = gi.num_lib.asarray(cov)

        num_particles = inputs.shape[1]
        log_priors, log_like = self._initialize_probabilities(inputs)

        for i in range(num_samples):

            inputs, log_like, log_priors, rejected = \
                 self.perform_mcmc_step(inputs, cov, log_like, log_priors, phi)

            cov = self._tune_covariance(num_particles, rejected, cov)

        with nvtx.annotate(message='get inputs, log_like', color='turquoise'):
            inputs = inputs.get()
            log_like = log_like.get()
        return inputs, log_like


    @staticmethod
    @nvtx.annotate(color='turquoise')
    def _tune_covariance(num_particles, rejected, cov):
        with nvtx.annotate(message='compute accepted', color='turquoise'):
            num_accepted = num_particles - gi.num_lib.sum(rejected, axis=1)

        scale_array = gi.num_lib.ones((cov.shape[0]))
        low_acceptance = (num_accepted < num_particles * 0.2)[:, 0]
        high_acceptance = (num_accepted > num_particles * 0.7)[:, 0]
        scale_array = gi.num_lib.where(low_acceptance, 0.2, scale_array)
        scale_array = gi.num_lib.where(high_acceptance, 2, scale_array)
        scale_array = scale_array.reshape(-1, 1, 1)

        cov *= scale_array
        return cov


    @abstractmethod
    def evaluate_model(self, inputs):
        '''
        Calls self._model with "inputs" and returns corresponding outputs.
        '''

    @nvtx.annotate(color='turquoise')
    def evaluate_log_priors(self, inputs):
        '''
        Assumes that all input parameters have prior U(-inf, inf) except the
        last column, which is assumed to be standard deviation of the noise and
        has prior U(0, inf).
        '''
        log_priors = gi.num_lib.zeros((inputs.shape[0], inputs.shape[1], 1))
        log_priors = gi.num_lib.where(inputs[:, :, -1:] < 0, -gi.num_lib.inf,
                                      log_priors)
        return log_priors

    @nvtx.annotate(color='turquoise')
    def evaluate_log_likelihood(self, inputs):
        return self._log_like_func(inputs)

    @staticmethod
    @nvtx.annotate(color='turquoise')
    def proposal(inputs, cov):
        chol = gi.num_lib.linalg.cholesky(cov)
        z = gi.num_lib.random.normal(0, 1, inputs.shape)
        delta = gi.num_lib.matmul(chol, gi.num_lib.transpose(z, (0, 2, 1)))
        return inputs + gi.num_lib.transpose(delta, (0, 2, 1))

    @staticmethod
    @nvtx.annotate(color='turquoise')
    def acceptance_ratio(new_log_like, old_log_like, new_log_priors,
                         old_log_priors):
        old_log_post = old_log_like + old_log_priors
        new_log_post = new_log_like + new_log_priors
        return gi.num_lib.exp(new_log_post - old_log_post)

    @staticmethod
    @nvtx.annotate(color='turquoise')
    def get_rejected(acceptance_ratios):
        u = gi.num_lib.random.uniform(0, 1, acceptance_ratios.shape)
        return acceptance_ratios < u

    @nvtx.annotate(color='turquoise')
    def _initialize_probabilities(self, inputs):
        log_priors = self.evaluate_log_priors(inputs)
        self._check_log_priors_for_zero_probability(log_priors)
        log_like = self.evaluate_log_likelihood(inputs)
        return log_priors, log_like

    @nvtx.annotate(color='turquoise')
    def perform_mcmc_step(self, inputs, cov, log_like, log_priors, phi):
        new_inputs = self.proposal(inputs, cov)
        new_log_priors = self.evaluate_log_priors(new_inputs)
        new_log_like = self.evaluate_log_likelihood(new_inputs)

        accpt_ratio = self.acceptance_ratio(new_log_like * phi,
                                            log_like * phi,
                                            new_log_priors, log_priors)

        rejected = self.get_rejected(accpt_ratio)

        inputs = gi.num_lib.where(rejected, inputs, new_inputs)
        log_like = gi.num_lib.where(rejected, log_like, new_log_like)
        log_priors = gi.num_lib.where(rejected, log_priors, new_log_priors)

        return inputs, log_like, log_priors, rejected

    @nvtx.annotate(color='turquoise')
    def _check_log_priors_for_zero_probability(self, log_priors):
        if (log_priors == -np.inf).any():
            raise ValueError('Initial inputs are out of bounds; '
                             f'prior log prob = {log_priors}')
