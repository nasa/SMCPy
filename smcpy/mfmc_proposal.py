import numpy as np
from log_likelihoods import Normal

class MultiFidelityProposal:
    """
    An empirical proposal distribution for Multi-Fidelity Sequential Monte Carlo (SMC).
    """

    def __init__(self, lofi_samples: np.ndarray, lofi_model, data: np.ndarray, 
                 noise_stdev: float, prior):
        """
        Initializes the empirical Multi-Fidelity proposal distribution.

        Args:
            lofi_samples (np.ndarray): Accepted posterior samples from the LF SMC run.
            lofi_model (callable): The LF model used to evaluate simulated outputs.
            data (np.ndarray): The true observed measurement data being targeted.
            noise_stdev (float): Standard deviation of the Gaussian noise.
            prior: The prior distribution object (must implement a `logpdf` method).
        """
        self.lofi_samples = lofi_samples
        self.lofi_model = lofi_model
        self.noisy_data = data
        self.noise_stdev = noise_stdev
        self.prior = prior

    def rvs(self, num_samples: int, random_state: int = None) -> np.ndarray:
        """
        Generates random samples directly from the Low-Fidelity posterior particles.
        """
        if num_samples == len(self.lofi_samples):
            return self.lofi_samples
            
        if random_state is not None:
            np.random.seed(random_state)

        random_indices = np.random.choice(self.lofi_samples.shape[0], size=num_samples, replace=False)
        return self.lofi_samples[random_indices]

    def logpdf(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluates the unnormalized log probability density of the proposed particles.
        Returns log(LF_Prior) + log(LF_Likelihood).
        """
        # 1. Evaluate the log-prior
        # Note: Make sure the prior's logpdf handles shapes consistently (e.g., returns shape (num_samples,))
        log_prior_values = self.prior.logpdf(inputs)
        
        # Reshape to (num_samples, 1) just in case
        log_prior_values = np.array(log_prior_values).reshape(-1, 1)
        
        # 2. Evaluate the LF log-likelihood
        log_likelihood = Normal(self.lofi_model, self.noisy_data, self.noise_stdev)
        log_L_values = log_likelihood(inputs)
        log_L_values = np.array(log_L_values).reshape(-1, 1)

        # 3. Sum them to get the unnormalized log LF posterior density
        # Mask out likelihood calculations if the prior is -inf to save computation? 
        # (Optional optimization, but standard addition works fine if Normal handles np.inf)
        log_posterior_values = log_prior_values + log_L_values

        return log_posterior_values