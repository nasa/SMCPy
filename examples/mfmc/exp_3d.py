import numpy as np
from typing import Union

resolution = 150
char_length = 4
# Sample 100 points for x and y ranging from -1 to 1
x = np.linspace(np.pi, 0, resolution)
y = np.linspace(np.pi, 0, resolution)

x = char_length * np.cos(x)
y = char_length * np.cos(y)

# Create a 2D meshgrid (100x100)
X, Y = np.meshgrid(x, y)

def M_HF(THETA: np.ndarray, return_flat: bool = True) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model over a 2D spatial grid.
    
    The model evaluates the function: Z = theta_0 * exp(x * y) + theta_1
    over a 100x100 grid where x and y range from -1 to 1.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters. 
            - Column 0 corresponds to theta_0 (the scaling factor).
            - Column 1 corresponds to theta_1 (the bias/offset).
        return_flat (bool, optional): If True, flattens the spatial dimensions of the 
            evaluated output array before returning. Defaults to True.

    Returns:
        np.ndarray: The evaluated high-fidelity output. If return_flat is True, the spatial 
        grid is flattened (1D per sample). Otherwise, it maintains the 2D grid shape. 
        The final shape also depends on the input N and NumPy broadcasting rules.
    """
    # Extract parameters, keeping the first dimension to allow batch processing of N samples
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Evaluate and return the exact exponential function
    if return_flat:
        return theta_0 * np.exp(X * Y).flatten() + theta_1
    else:
        return theta_0 * np.exp(X * Y) + theta_1

def M_LF8(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF) model using a Maclaurin series approximation.
    
    This approximates the high-fidelity exponential term `exp(x * y)` using an 
    8th-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
            - Column 0 corresponds to theta_0.
            - Column 1 corresponds to theta_1.

    Returns:
        np.ndarray: The evaluated low-fidelity approximation, flattened to a 1D array per sample.
    """
    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Calculate the joint xy term once to optimize computation
    xy = X * Y  
    
    # Calculate the 8th degree Maclaurin series approximation
    # Z = theta_0 * (1 + xy + xy^2/2! + xy^3/3! + ... + xy^8/8!) + theta_1
    return theta_0 * (
        1 + 
        (xy) + 
        (xy**2 / 2) + 
        (xy**3 / 6) +
        (xy**4 / 24) +
        (xy**5 / 120) +
        (xy**6 / 720) +
        (xy**7 / 5040) +
        (xy**8 / 40320)
    ).flatten() + theta_1

def M_LF4(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF4) model using a truncated Maclaurin series.
    
    This approximates the high-fidelity exponential term `exp(x * y)` using a 
    4th-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
            - Column 0 corresponds to theta_0.
            - Column 1 corresponds to theta_1.

    Returns:
        np.ndarray: The evaluated low-fidelity approximation, flattened to a 1D array per sample.
    """
    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Calculate the joint xy term once to optimize computation
    xy = X * Y  
    
    # Calculate the 4th degree Maclaurin series approximation
    # Z = theta_0 * (1 + xy + xy^2/2! + xy^3/3! + xy^4/4!) + theta_1
    return theta_0 * (
        1 + 
        (xy) + 
        (xy**2 / 2) + 
        (xy**3 / 6) + 
        (xy**4 / 24)
    ).flatten() + theta_1

def M_LF6(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF6) model using a truncated Maclaurin series.
    
    This approximates the high-fidelity exponential term `exp(x * y)` using a 
    6th-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
            - Column 0 corresponds to theta_0.
            - Column 1 corresponds to theta_1.

    Returns:
        np.ndarray: The evaluated low-fidelity approximation, flattened to a 1D array per sample.
    """
    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Calculate the joint xy term once to optimize computation
    xy = X * Y  
    
    # Calculate the 6th degree Maclaurin series approximation
    # Z = theta_0 * (1 + xy + xy^2/2! + xy^3/3! + ... + xy^6/6!) + theta_1
    return theta_0 * (
        1 + 
        (xy) + 
        (xy**2 / 2) + 
        (xy**3 / 6) + 
        (xy**4 / 24) +
        (xy**5 / 120) +
        (xy**6 / 720)
    ).flatten() + theta_1

def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, return_flat: bool = True, random_seed: int = None) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the true model parameters.
        noise_st_dev (float): The standard deviation of the Gaussian noise to be added.
        return_flat (bool, optional): If True, flattens the output array before returning. 
            Defaults to True.
        random_seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: The simulated noisy data.
    """
    # Run the "true" high-fidelity simulation
    Z_HF = M_HF(THETA)
    
    # Use a local random number generator instead of messing with the global np.random state
    rng = np.random.default_rng(random_seed)
    
    # Generate Gaussian (normal) noise
    random_noise = rng.normal(0, noise_st_dev, size=Z_HF.shape)

    # Add noise to the true signal
    noisy_data = random_noise + Z_HF
    
    if return_flat:
        return noisy_data.flatten()
    else:
        return noisy_data