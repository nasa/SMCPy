import numpy as np
from typing import Any
from log_likelihoods import Normal

class MultiFidelityProposal:
    """
    A helper class to combine an arbitrary number of independent proposal 
    distributions into a single multivariate distribution object.
    
    This is intended for use with the `smcpy.paths` module. It exposes 
    standard `scipy.stats`-like methods (`rvs` and `logpdf`) to sample from 
    and evaluate the joint probability of the combined independent distributions.
    """

    def __init__(self, lofi_samples, lofi_model, data, *args: Any):
        """
        Initializes the multivariate independent distribution.

        Args:
            *args: An arbitrary number of `scipy.stats`-like distribution objects. 
                Each object must implement `rvs()` and `logpdf()` methods.
        """

        self.lofi_samples = lofi_samples
        self.lofi_model = lofi_model
        self.noisy_data = data

        self._dist_list = args
        self._dims = self._get_dims()

    def rvs(self, num_samples: int, random_state: int = None) -> np.ndarray:
        """
        Generates random samples from the joint multivariate distribution.

        Args:
            num_samples (int): The number of samples to generate.
            random_state (int, optional): A seed for the random number generator 
                to ensure reproducibility. Defaults to None.

        Returns:
            np.ndarray: A 2D array of shape `(num_samples, total_dimensions)` 
                containing the generated samples.
        """
        # Take a sample from the low fidelity posterior

        # return this sample
        return None

    def logpdf(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluates the joint log probability density function (logpdf) for the inputs.

        Because the distributions are independent, the joint logpdf is the sum 
        of the individual logpdfs.

        Args:
            inputs (np.ndarray): A 2D array of shape `(num_samples, total_dimensions)` 
                representing the input states to evaluate.

        Returns:
            np.ndarray: A 2D array of shape `(num_samples, 1)` containing the 
                evaluated joint log probabilities.
        """

        # Compute log-likelihood of low fidelity data
        lofi_log_likelihoods = Normal(self.lofi_model, self.noisy_data, self.noise_stdev)

        # Pair each distribution with its corresponding slice of the input array
        iterable = zip(self._dist_list, self._partition_inputs(inputs))
        
        # Evaluate logpdf for each independent distribution and stack them as columns
        indv_logpdf = np.column_stack([d.logpdf(in_) for d, in_ in iterable])
        
        # Sum the independent log probabilities across the dimension axis
        return np.sum(indv_logpdf, axis=1, keepdims=True)

    def _get_dims(self) -> list:
        """
        Determines the dimensionality of each independent distribution.

        Returns:
            list: A list of integers representing the number of dimensions for 
                each distribution.
        """
        # Samples 1 value from each distribution to determine its size/dimensionality
        return [d.rvs(1).size for d in self._dist_list]

    def _partition_inputs(self, inputs: np.ndarray) -> list:
        """
        Splits the aggregated input array into chunks corresponding to the 
        dimensions of the individual distributions.

        Args:
            inputs (np.ndarray): The full 2D input array.

        Returns:
            list: A list of sub-arrays, each mapping to an individual distribution.
        """
        # Split inputs based on the cumulative sum of the dimensions
        split_indices = np.cumsum(self._dims)[:-1]
        return np.split(inputs, split_indices, axis=1)