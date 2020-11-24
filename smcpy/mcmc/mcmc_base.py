import numpy as np

from abc import ABC, abstractmethod
from tqdm import tqdm

from .mcmc_logger import MCMCLogger


class MCMCBase(ABC, MCMCLogger):
    
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
        self._log_like_args = log_like_args
        self._log_like_func = log_like_func

        super().__init__(__name__, debug)

    def sample_from_priors(self, num_samples):
        samples = [p.rvs(num_samples).reshape(-1, 1) \
                   for i, p in enumerate(self._priors)]
        return np.hstack(samples)

    def evaluate_log_priors(self, inputs):
        priors = np.hstack([p.pdf(inputs.T[i]).reshape(-1, 1) \
                            for i, p in enumerate(self._priors)])
        nonzero_priors = priors != 0
        log_priors = np.ones(priors.shape) * -np.inf
        log_priors = np.log(priors, where=nonzero_priors, out=log_priors)
        return log_priors

    @abstractmethod
    def evaluate_model(self, inputs):
        '''
        Calls self._model with "inputs" and returns corresponding outputs.
        '''

    def evaluate_log_likelihood(self, inputs):
        log_like = self._log_like_func(inputs, self.evaluate_model, self._data,
                                       self._log_like_args)
        return log_like.reshape(-1, 1)

    @staticmethod
    def evaluate_log_posterior(log_likelihood, log_priors):
        return np.sum(np.hstack((log_likelihood, log_priors)), axis=1)

    @staticmethod
    def proposal(inputs, cov):
        scale_factor = 1 #2.38 ** 2 / cov.shape[0] # From Smith 2014, pg. 172
        mean = np.zeros(cov.shape[0])
        delta = np.random.multivariate_normal(mean, scale_factor * cov,
                                              inputs.shape[0])
        return inputs + delta

    def acceptance_ratio(self, new_log_like, old_log_like, new_log_priors,
                         old_log_priors):
        old_log_post = self.evaluate_log_posterior(old_log_like, old_log_priors)
        new_log_post = self.evaluate_log_posterior(new_log_like, new_log_priors)
        return np.exp(new_log_post - old_log_post).reshape(-1, 1)

    @staticmethod
    def selection(new_values, old_values, acceptance_ratios, u):
        reject = acceptance_ratios < u
        return np.where(reject, old_values, new_values)

    @staticmethod
    def adapt_proposal_cov(cov, chain, sample_idx, adapt_interval, adapt_delay):
        if adapt_interval is None or sample_idx < adapt_delay - 1:
            return cov

        start = 0
        if sample_idx > adapt_delay + adapt_interval:
            start = adapt_delay

        if adapt_interval <= sample_idx and sample_idx % adapt_interval == 0:
            flat_chain = [chain[:, i, start:sample_idx + 2].flatten() \
                          for i in range(chain.shape[1])]
            cov = np.cov(flat_chain)

        return cov

    def smc_metropolis(self, inputs, num_samples, cov, phi):
        log_priors = self.evaluate_log_priors(inputs)
        self._check_log_priors_for_zero_probability(log_priors)
        log_like = self.evaluate_log_likelihood(inputs)
    
        for i in range(num_samples):

            new_inputs = self.proposal(inputs, cov)
            new_log_priors = self.evaluate_log_priors(new_inputs)
            new_log_like = self._eval_log_like_if_prior_nonzero(new_log_priors,
                                                                new_inputs)

            accpt_ratio = self.acceptance_ratio(new_log_like * phi,
                                                log_like * phi,
                                                new_log_priors, log_priors)

            u = np.random.uniform(0, 1, accpt_ratio.shape)
    
            inputs = self.selection(new_inputs, inputs, accpt_ratio, u)
            log_like = self.selection(new_log_like, log_like, accpt_ratio, u)
            log_priors = self.selection(new_log_priors, log_priors,
                                        accpt_ratio, u)
    
        return inputs, log_like

    def metropolis(self, inputs, num_samples, cov, adapt_interval=None,
                   adapt_delay=0, progress_bar=False, **kwargs):
        chain = np.zeros([inputs.shape[0], inputs.shape[1], num_samples + 1])
        chain[:, :, 0] = inputs

        log_priors = self.evaluate_log_priors(inputs)
        self._check_log_priors_for_zero_probability(log_priors)
        log_like = self.evaluate_log_likelihood(inputs)

        self._write_sample_to_log(inputs, log_like, log_priors, 0, False)
    
        for i in tqdm(range(num_samples), disable=not progress_bar):
    
            new_inputs = self.proposal(inputs, cov)
            new_log_priors = self.evaluate_log_priors(new_inputs)
            new_log_like = self._eval_log_like_if_prior_nonzero(new_log_priors,
                                                                new_inputs)

            self._write_sample_to_log(new_inputs, new_log_like, new_log_priors,
                                      i, True)

            accpt_ratio = self.acceptance_ratio(new_log_like, log_like,
                                                new_log_priors, log_priors)

            u = np.random.uniform(0, 1, accpt_ratio.shape)
    
            inputs = self.selection(new_inputs, inputs, accpt_ratio, u)
            log_like = self.selection(new_log_like, log_like, accpt_ratio, u)
            log_priors = self.selection(new_log_priors, log_priors,
                                        accpt_ratio, u)

            self._write_accpt_to_log(accpt_ratio, u)

            chain[:, :, i + 1] = inputs

            cov = self.adapt_proposal_cov(cov, chain, i, adapt_interval,
                                          adapt_delay)

            self._write_cov_to_log(cov)

        return chain

    def _check_log_priors_for_zero_probability(self, log_priors):
        if (log_priors == -np.inf).any():
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
