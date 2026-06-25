import numpy as np

def generate_noisy_data(theta_true: float, noise_st_dev: float, num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generates synthetic observation data y = theta + epsilon.

    Args:
        theta_true: The true underlying parameter value.
        noise_st_dev: The standard deviation of the Gaussian noise (epsilon).
        num_samples: The number of data points to generate (N).
        random_seed: Seed for reproducibility.

    Returns:
        np.ndarray: A 1D array of length `num_samples` containing the observations.
    """
    rng = np.random.default_rng(random_seed)
    epsilon = rng.normal(0, noise_st_dev, size=num_samples)
    return theta_true + epsilon

def M_LF(THETA: np.ndarray, num_lf_samples: int) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF) model.

    In this conjugate Gaussian problem, the model simply predicts that all observations
    equal the parameter theta. The "fidelity" is governed by how many samples
    it predicts (which corresponds to how much data the likelihood function will evaluate).

    Args:
        THETA: A 2D array of shape (M, 1) containing the model parameters (particles).
        num_lf_samples: The number of observations the LF model processes (small N).

    Returns:
        np.ndarray: An array of shape (M, num_lf_samples) containing the predictions.
    """
    # Tile the THETA column to match the number of LF data points
    return np.tile(THETA, (1, num_lf_samples))

def M_HF(THETA: np.ndarray, num_hf_samples: int) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model.

    Similar to the LF model, but evaluates against a much larger number of samples,
    creating a sharper, less biased likelihood landscape.

    Args:
        THETA: A 2D array of shape (M, 1) containing the model parameters (particles).
        num_hf_samples: The number of observations the HF model processes (large N).

    Returns:
        np.ndarray: An array of shape (M, num_hf_samples) containing the predictions.
    """
    return np.tile(THETA, (1, num_hf_samples))

def get_analytical_posterior(
    data: np.ndarray,
    data_std: float,
    prior_mean: float,
    prior_std: float
) -> tuple:
    """
    Calculates the exact analytical posterior for a conjugate Normal-Normal model.

    Args:
        data: The observed data array.
        data_std: The known standard deviation of the likelihood noise.
        prior_mean: The mean of the Normal prior.
        prior_std: The standard deviation of the Normal prior.

    Returns:
        tuple: (posterior_mean, posterior_std)
    """
    N = len(data)
    y_bar = np.mean(data)

    prior_var = prior_std ** 2
    data_var = data_std ** 2

    post_precision = (1.0 / prior_var) + (N / data_var)
    post_var = 1.0 / post_precision

    post_mean = post_var * ((prior_mean / prior_var) + ((N * y_bar) / data_var))

    return post_mean, np.sqrt(post_var)
