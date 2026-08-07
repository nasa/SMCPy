import numpy as np
from typing import Union

def _generate_meshgrid(char_length: int, resolution: int = 150):
    """
    Helper function to generate the 2D spatial meshgrid based on char_length.

    Note that our grid here more densely samples the edges of the domain than the center 
    due to the nature of the exponential function.
    """
    x = np.linspace(np.pi, 0, resolution)
    y = np.linspace(np.pi, 0, resolution)

    x = char_length * np.cos(x)
    y = char_length * np.cos(y)

    return np.meshgrid(x, y)

def M_HF(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model over a 2D spatial grid.
    
    The model evaluates the function: Z = theta_0 * exp(x * y) + theta_1

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
        return_flat (bool, optional): If True, flattens the spatial dimensions. Defaults to True.
        char_length (int, optional): Scaling factor for the spatial domain. Defaults to 4.

    Returns:
        np.ndarray: The evaluated high-fidelity output.
    """
    X, Y = _generate_meshgrid(char_length)

    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    return theta_0 * np.exp(X * Y).flatten() + theta_1


def M_LF(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity model using a 5th-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6) + (xy**4 / 24) +
        (xy**5 / 120)
    ).flatten() + theta_1

def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, random_seed: int = None, char_length: int = 2) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.

    Args:
        THETA (np.ndarray): A 2D array containing the true model parameters.
        noise_st_dev (float): Standard deviation of the Gaussian noise.
        return_flat (bool, optional): Flattens the output array. Defaults to True.
        random_seed (int, optional): Seed for reproducibility. Defaults to None.
        char_length (int, optional): Scaling factor for the spatial domain. Defaults to 4.
    """
    # Run the "true" high-fidelity simulation
    Z_HF = M_HF(THETA, char_length=char_length)
    
    # Generate reproducible noise
    rng = np.random.default_rng(random_seed)
    random_noise = rng.normal(0, noise_st_dev, size=Z_HF.shape)

    # Add noise to the true signal
    noisy_data = random_noise + Z_HF
    
    return noisy_data.flatten()