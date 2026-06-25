import numpy as np
from typing import Union

resolution = 10_000
char_length = 6

# Generate 1D uniform grid, range: [-4, 4]
x = np.linspace(-char_length, char_length, resolution)

def M_HF(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model over a 1D spatial domain.
    
    The model evaluates the function: Z = theta_0 * sin(x) + theta_1
    over a 1D array of 10,000 points ranging from -4 to 4.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters. 
            - Column 0 corresponds to theta_0 (the scaling factor).
            - Column 1 corresponds to theta_1 (the bias/offset).

    Returns:
        np.ndarray: The evaluated high-fidelity output with shape (N, resolution).
    """
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    output = theta_0 * np.sin(x) + theta_1
    
    return output

def M_LF13(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF13) model using a truncated Maclaurin series.
    
    This approximates the high-fidelity sine term `sin(x)` using a 
    13th-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2).

    Returns:
        np.ndarray: The evaluated 13th-degree approximation with shape (N, resolution).
    """
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Z = theta_0 * (x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11! + x^13/13!) + theta_1
    output = theta_0 * (
        x - 
        (x**3 / 6) + 
        (x**5 / 120) - 
        (x**7 / 5040) + 
        (x**9 / 362880) -
        (x**11 / 39916800) +
        (x**13 / 6227020800)
    ) + theta_1
    
    return output

def M_LF9(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF9) model using a truncated Maclaurin series.
    
    This approximates the high-fidelity sine term `sin(x)` using a 
    9th-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2).

    Returns:
        np.ndarray: The evaluated 9th-degree approximation with shape (N, resolution).
    """
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Z = theta_0 * (x - x^3/3! + x^5/5! - x^7/7! + x^9/9!) + theta_1
    output = theta_0 * (
        x - 
        (x**3 / 6) + 
        (x**5 / 120) - 
        (x**7 / 5040) + 
        (x**9 / 362880)
    ) + theta_1
    
    return output

def M_LF3(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the lowest-fidelity (LF3) model using a truncated Maclaurin series.
    
    This approximates the high-fidelity sine term `sin(x)` using a 
    3rd-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2).

    Returns:
        np.ndarray: The evaluated 3rd-degree approximation with shape (N, resolution).
    """
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    # Z = theta_0 * (x - x^3/3!) + theta_1
    output = theta_0 * (
        x - 
        (x**3 / 6)
    ) + theta_1
    
    return output

def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, random_seed: int = None) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the true model parameters.
        noise_st_dev (float): The standard deviation of the Gaussian noise to be added.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: The simulated noisy data with shape (N, resolution).
    """
    Z_HF = M_HF(THETA)
    
    rng = np.random.default_rng(random_seed)
    random_noise = rng.normal(0, noise_st_dev, size=Z_HF.shape)

    noisy_data = Z_HF + random_noise
    
    return noisy_data