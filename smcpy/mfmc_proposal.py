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
                 lofi_particles: np.ndarray, 
                 lofi_model: Callable, 
                 observed_data: np.ndarray, 
                 priors: List[Any],
                 additive_noise_stdev: Optional[float] = None):
        """
        Initializes the Multi-Fidelity proposal distribution.

        Args:
            lofi_particles (np.ndarray): Accepted posterior samples from the LF SMC run.
            lofi_model (Callable): The LF model used to evaluate simulated outputs.
            observed_data (np.ndarray): The true observed measurement data being targeted.
            priors (List[Any]): List of prior distribution objects for the parameters.
            additive_noise_stdev (Optional[float]): Standard deviation of the Gaussian noise.
        """
        self.lofi_particles = lofi_particles
        self.lofi_model = lofi_model
        self.observed_data = observed_data
        self.additive_noise_stdev = additive_noise_stdev
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
        if num_samples == len(self.lofi_particles):
            return self.lofi_particles
            
        if random_state is not None:
            np.random.seed(random_state)

        random_indices = np.random.choice(self.lofi_particles.shape[0], size=num_samples, replace=False)
        return self.lofi_particles[random_indices]

    def logpdf(self, particles: np.ndarray) -> np.ndarray:
        """
        Evaluates the unnormalized log probability density of the proposed particles.

        Args:
            particles (np.ndarray): Proposed parameter sets of shape (num_samples, N_parameters).

        Returns:
            np.ndarray: Unnormalized log LF posterior density of shape (num_samples, 1).
        """
        # 1. Evaluate the log-prior
        log_prior_values = self.prior_logpdf(particles)
        log_prior_values = np.array(log_prior_values).reshape(-1, 1)
        
        # 2. Evaluate the LF log-likelihood
        log_likelihood = Normal(self.lofi_model, self.observed_data, self.additive_noise_stdev)
        log_likelihood_values = np.array(log_likelihood(particles)).reshape(-1, 1)

        # 3. Sum prior and likelihood to get the unnormalized log LF posterior density
        return log_prior_values + log_likelihood_values
    
    def prior_logpdf(self, particles: np.ndarray) -> np.ndarray:
        """
        Computes the joint log-prior probability for the particles.

        Args:
            particles (np.ndarray): Parameter sets to evaluate.

        Returns:
            np.ndarray: Joint log-prior probabilities of shape (num_samples, 1).
        """
        iterable = zip(self._priors_list, self._partition_inputs(particles))
        marginal_logpdfs = np.hstack([prior.logpdf(partial_particle) for prior, partial_particle in iterable])
        return np.sum(marginal_logpdfs, axis=1, keepdims=True)
    
    def _get_dims(self) -> List[int]:
        """
        Determines the dimensionality of each prior distribution.

        Returns:
            List[int]: A list of parameter dimension sizes.
        """
        return [prior.rvs(1).size for prior in self._priors_list]

    def _partition_inputs(self, particles: np.ndarray) -> List[np.ndarray]:
        """
        Splits the input array along columns to match individual prior dimensions.

        Args:
            particles (np.ndarray): Full parameter array.

        Returns:
            List[np.ndarray]: A list of sub-arrays corresponding to each prior.
        """
        return np.split(particles, np.cumsum(self._dims)[:-1], axis=1)