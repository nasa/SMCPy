import numpy as np

from abc import ABC, abstractmethod
from tqdm import tqdm

import smcpy.utils.global_imports as gi

class MCMCBase(ABC):

    def __init__(self, model, data, priors, log_like_args, log_like_func,
                 debug):
        '''
        :param model: maps inputs to outputs
        :type model: callable
        :param data: data corresponding to model outputs
        :type data: 1D array
        :param priors: random variable objects with a pdf and rvs method (e.g.
            scipy stats random variable objects)
        :type priors: list of objects
        :param log_like_args: any fixed parameters that define the likelihood
            function (e.g., standard deviation for a Gaussian likelihood).
        :type log_like_args: 1D array or None
        :param log_like_func: log likelihood function that takes inputs, model,
            data, and hyperparameters and returns log likelihoods
        :type log_like_func: callable
        '''
        self._eval_model = model
        self._data = data
        self._priors = priors
        self._log_like_func = log_like_func(self.evaluate_model, data,
                                            log_like_args)

    def smc_metropolis(self, inputs, num_samples, cov, phi):
        num_particles = inputs.shape[0]
        log_priors, log_like = self._initialize_probabilities(inputs)

        #if gi.USING_GPU:
        #    cov = gi.num_lib.asarray(cov)

        for i in range(num_samples):

            inputs, log_like, log_priors, rejected = \
                self._perform_mcmc_step(inputs, cov, log_like, log_priors, phi)

            num_accepted = num_particles - np.sum(rejected)

            if num_accepted < inputs.shape[0] * 0.2:
                cov = cov * 1 / 5
            if num_accepted > inputs.shape[0] * 0.7:
                cov = cov * 2

        return inputs, log_like

    def metropolis(self, inputs, num_samples, cov, adapt_interval=None,
                   adapt_delay=0, progress_bar=False, **kwargs):

        chain = np.zeros([inputs.shape[0], inputs.shape[1], num_samples + 1])
        chain[:, :, 0] = inputs

        log_priors, log_like = self._initialize_probabilities(inputs)

        for i in tqdm(range(1, num_samples + 1), disable=not progress_bar):
            inputs, log_like, log_priors, rejected = \
                self._perform_mcmc_step(inputs, cov, log_like, log_priors, 1)
            chain[:, :, i] = inputs

            cov = self.adapt_proposal_cov(cov, chain, i, adapt_interval,
                                          adapt_delay)
        return chain

    @abstractmethod
    def evaluate_model(self, inputs):
        '''
        Calls self._model with "inputs" and returns corresponding outputs.
        '''

    def sample_from_priors(self, num_samples):
        samples = []
        for i, p in enumerate(self._priors):
            samples.append(p.rvs(num_samples).T)
        return np.vstack(samples).T

    def evaluate_log_priors(self, inputs):
        prior_dims = self._get_prior_dims()

        if inputs.shape[1] != sum(prior_dims):
            raise ValueError("Num prior distributions != num input params")

        prior_probs = np.empty((inputs.shape[0], len(self._priors)))
        in_start_idx = 0
        for i, p in enumerate(self._priors):
            in_ = inputs[:, in_start_idx:in_start_idx + prior_dims[i]]
            prior_probs[:, i] = p.pdf(in_).squeeze()
            in_start_idx += prior_dims[i]

        nonzero_priors = prior_probs != 0
        log_priors = np.full(prior_probs.shape, -np.inf)
        log_priors = np.log(prior_probs, where=nonzero_priors, out=log_priors)
        return log_priors

    def _get_prior_dims(self):
        return [p.dim if hasattr(p, 'dim') else 1 for p in self._priors]

    def evaluate_log_likelihood(self, inputs):
        log_like = self._log_like_func(inputs)
        return log_like.reshape(-1, 1)

    @staticmethod
    def evaluate_log_posterior(log_likelihood, log_priors):
        return np.sum(np.hstack((log_likelihood, log_priors)), axis=1)

    @staticmethod
    def proposal(inputs, cov):
        scale_factor = 1  # 2.38 ** 2 / cov.shape[0] # From Smith 2014, pg. 172
        cov *= scale_factor
        chol = np.linalg.cholesky(cov)
        z = np.random.normal(0, 1, inputs.shape)
        delta = np.matmul(chol, z.T).T
        return inputs + delta

    def acceptance_ratio(self, new_log_like, old_log_like, new_log_priors,
                         old_log_priors):
        old_log_post = self.evaluate_log_posterior(old_log_like, old_log_priors)
        new_log_post = self.evaluate_log_posterior(new_log_like, new_log_priors)
        return np.exp(new_log_post - old_log_post).reshape(-1, 1)

    @staticmethod
    def get_rejections(acceptance_ratios):
        u = np.random.uniform(0, 1, acceptance_ratios.shape)
        return acceptance_ratios < u

    def adapt_proposal_cov(self, cov, chain, idx, adapt_interval, adapt_delay):

        if self._is_adapt_iteration(adapt_interval, idx, adapt_delay):
            start = self._get_window_start(idx, adapt_delay, adapt_interval)
            end = idx + 1
            n_param = chain.shape[1]
            flat_chain = [chain[:, i, start:end].flatten() \
                          for i in range(n_param)]
            return np.cov(flat_chain)
        return cov

    def _initialize_probabilities(self, inputs):
        log_priors = self.evaluate_log_priors(inputs)
        self._check_log_priors_for_zero_probability(log_priors)
        log_like = self.evaluate_log_likelihood(inputs)
        return log_priors, log_like

    def _perform_mcmc_step(self, inputs, cov, log_like, log_priors, phi):
        new_inputs = self.proposal(inputs, cov)
        new_log_priors = self.evaluate_log_priors(new_inputs)
        new_log_like = self._eval_log_like_if_prior_nonzero(new_log_priors,
                                                            new_inputs)

        accpt_ratio = self.acceptance_ratio(new_log_like * phi,
                                            log_like * phi,
                                            new_log_priors, log_priors)

        rejected = self.get_rejections(accpt_ratio)

        inputs = np.where(rejected, inputs, new_inputs)
        log_like = np.where(rejected, log_like, new_log_like)
        log_priors = np.where(rejected, log_priors, new_log_priors)

        return inputs, log_like, log_priors, rejected

    @staticmethod
    def _is_adapt_iteration(adapt_interval, idx, adapt_delay):
        if adapt_interval is None:
            return False
        surpassed_delay = idx >= adapt_delay
        is_adapt_iteration = (idx - adapt_delay) % adapt_interval == 0
        return surpassed_delay and is_adapt_iteration

    @staticmethod
    def _get_window_start(idx, adapt_delay, adapt_interval):
        if idx >= adapt_delay + adapt_interval:
            return adapt_delay + 1
        return max(adapt_delay - adapt_interval + 1, 1)

    def _check_log_priors_for_zero_probability(self, log_priors):
        if any(~self._row_has_nonzero_prior_probability(log_priors)):
            raise ValueError('Initial inputs are out of bounds; '
                             f'prior log prob = {log_priors}')

    def _eval_log_like_if_prior_nonzero(self, log_priors, inputs):
        pos_rows = self._row_has_nonzero_prior_probability(log_priors)
        log_likes = np.zeros((log_priors.shape[0], 1))
        if inputs[pos_rows].size != 0:
            log_likes[pos_rows] = self.evaluate_log_likelihood(inputs[pos_rows])
        return log_likes

    @staticmethod
    def _row_has_nonzero_prior_probability(log_priors):
        return ~(log_priors == -np.inf).any(axis=1)
