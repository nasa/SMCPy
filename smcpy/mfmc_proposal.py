import numpy as np
from typing import List, Callable, Optional, Any
from log_likelihoods import Normal

class MultiFidelityProposal:
    """
    An empirical proposal distribution for Multi-Fidelity Sequential Monte Carlo (SMC).
    
    This class utilizes samples from a low-fidelity (LF) posterior to act as a 
    proposal distribution for a high-fidelity SMC run.
    """

    def __init__(self, 
                 lofi_samples: np.ndarray, 
                 lofi_model: Callable, 
                 observed_data: np.ndarray, 
                 noise_stdev: float, 
                 priors: List[Any]):
        """
        Initializes the Multi-Fidelity proposal distribution.

        Args:
            lofi_samples (np.ndarray): Accepted posterior samples from the LF SMC run.
            lofi_model (Callable): The LF model used to evaluate simulated outputs.
            data (np.ndarray): The true observed measurement data being targeted.
            noise_stdev (float): Standard deviation of the Gaussian noise.
            prior (List[Any]): List of prior distribution objects for the parameters.
        """
        self.lofi_samples = lofi_samples
        self.lofi_model = lofi_model
        self.observed_data = observed_data
        self.noise_stdev = noise_stdev
        self._priors_list = priors
        self._dims = self._get_dims()

    def rvs(self, num_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generates random samples directly from the LF posterior particles.

        Args:
            num_samples (int): Number of particles to propose.
            random_state (Optional[int]): Seed for reproducibility.

        Returns:
            np.ndarray: Subsampled particles of shape (num_samples, N_parameters).
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

        Args:
            inputs (np.ndarray): Proposed parameter sets of shape (num_samples, N_parameters).

        Returns:
            np.ndarray: Unnormalized log LF posterior density of shape (num_samples, 1).
        """
        # 1. Evaluate the log-prior
        log_prior_values = self.prior_logpdf(inputs)
        log_prior_values = np.array(log_prior_values).reshape(-1, 1)
        
        # 2. Evaluate the LF log-likelihood
        log_likelihood = Normal(self.lofi_model, self.observed_data, self.noise_stdev)
        log_L_values = np.array(log_likelihood(inputs)).reshape(-1, 1)

        # 3. Sum prior and likelihood to get the unnormalized log LF posterior density
        return log_prior_values + log_L_values
    
    def prior_logpdf(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the joint log-prior probability for the inputs.

        Args:
            inputs (np.ndarray): Parameter sets to evaluate.

        Returns:
            np.ndarray: Joint log-prior probabilities of shape (num_samples, 1).
        """
        iterable = zip(self._priors_list, self._partition_inputs(inputs))
        indv_logpdf = np.hstack([d.logpdf(in_) for d, in_ in iterable])
        return np.sum(indv_logpdf, axis=1, keepdims=True)
    
    def _get_dims(self) -> List[int]:
        """
        Determines the dimensionality of each prior distribution.

        Returns:
            List[int]: A list of parameter dimension sizes.
        """
        return [d.rvs(1).size for d in self._priors_list]

    def _partition_inputs(self, inputs: np.ndarray) -> List[np.ndarray]:
        """
        Splits the input array along columns to match individual prior dimensions.

        Args:
            inputs (np.ndarray): Full parameter array.

        Returns:
            List[np.ndarray]: A list of sub-arrays corresponding to each prior.
        """
        return np.split(inputs, np.cumsum(self._dims)[:-1], axis=1)