import numpy as np

class VectorMCMC:
    
    def __init__(self, model, data, priors, std_dev=None):
        self._model = model
        self._data = data
        self._priors = priors
        self._fixed_std_dev = std_dev

    def sample_from_priors(self, num_samples):
        samples = [p.rvs(num_samples).reshape(-1, 1) \
                   for i, p in enumerate(self._priors)]
        return np.hstack(samples)

    def evaluate_log_priors(self, inputs):
        log_priors = [np.log(p.pdf(inputs.T[i]).reshape(-1, 1)) \
                      for i, p in enumerate(self._priors)]
        return np.hstack(log_priors)

    def evaluate_log_likelihood(self, inputs):
        std_dev = self._fixed_std_dev
        if std_dev is None:
            std_dev = inputs[:, -1]
        var = std_dev ** 2

        output = self._model(inputs)
        data = np.tile(self._data, [inputs.shape[0], 1])
        ssqe = np.sum((output - data) ** 2, axis=1)

        term1 = -np.log(2 * np.pi * var) * (data.shape[1] / 2.) 
        term2 = -1 / 2. * ssqe / var
        return (term1 + term2).reshape(-1, 1)

    @staticmethod
    def evaluate_log_posterior(log_likelihood, log_priors):
        return np.sum(np.hstack((log_likelihood, log_priors)), axis=1)

    @staticmethod
    def proposal(inputs, cov):
        scale_factor = 2.38 ** 2 / cov.shape[0] # From Smith 2014, pg. 172
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
    def adapt_proposal_cov(cov, chain, sample_count, adapt_interval):
        if adapt_interval is not None and sample_count % adapt_interval == 0:
            flat_chain = [chain[:, i, :sample_count + 2].flatten() \
                          for i in range(chain.shape[1])]
            cov = np.cov(flat_chain)
        return cov

    def smc_metropolis(self, inputs, num_samples, cov, phi):
        log_like = self.evaluate_log_likelihood(inputs)
        log_priors = self.evaluate_log_priors(inputs)
    
        for i in range(num_samples):

            new_inputs = self.proposal(inputs, cov)
            new_log_like = self.evaluate_log_likelihood(new_inputs)
            new_log_priors = self.evaluate_log_priors(new_inputs)
    
            accpt_ratio = self.acceptance_ratio(new_log_like * phi,
                                                log_like * phi,
                                                new_log_priors, log_priors)

            u = np.random.uniform(0, 1, accpt_ratio.shape)
    
            inputs = self.selection(new_inputs, inputs, accpt_ratio, u)
            log_like = self.selection(new_log_like, log_like, accpt_ratio, u)
            log_priors = self.selection(new_log_priors, log_priors,
                                        accpt_ratio, u)
    
        return inputs, log_like

    def metropolis(self, inputs, num_samples, cov, adapt_interval=None):
        chain = np.zeros([inputs.shape[0], inputs.shape[1], num_samples + 1])
        chain[:, :, 0] = inputs

        log_like = self.evaluate_log_likelihood(inputs)
        log_priors = self.evaluate_log_priors(inputs)
    
        for i in range(num_samples):
    
            new_inputs = self.proposal(inputs, cov)
            new_log_like = self.evaluate_log_likelihood(new_inputs)
            new_log_priors = self.evaluate_log_priors(new_inputs)
    
            accpt_ratio = self.acceptance_ratio(new_log_like, log_like,
                                                new_log_priors, log_priors)

            u = np.random.uniform(0, 1, accpt_ratio.shape)
    
            inputs = self.selection(new_inputs, inputs, accpt_ratio, u)
            log_like = self.selection(new_log_like, log_like, accpt_ratio, u)
            log_priors = self.selection(new_log_priors, log_priors,
                                        accpt_ratio, u)

            chain[:, :, i + 1] = inputs

            cov = self.adapt_proposal_cov(cov, chain, i, adapt_interval)
        print(cov)
    
        return chain
