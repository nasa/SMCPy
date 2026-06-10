import numpy as np
from typing import Union

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

    char_length = 2
    
    # Sample 100 points for x and y ranging from -1 to 1
    x = np.linspace(-1 * char_length, char_length, 100)
    y = np.linspace(-1 * char_length, char_length, 100)

    # Create a 2D meshgrid (100x100) for the spatial coordinates
    X, Y = np.meshgrid(x, y)

    # Evaluate and return the exact exponential function
    if return_flat:
        return theta_0 * np.exp(X * Y).flatten() + theta_1
    else:
        return theta_0 * np.exp(X * Y) + theta_1

def M_LF(THETA: np.ndarray, return_flat: bool = True) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF) model using a Maclaurin series approximation.
    
    This approximates the high-fidelity exponential term `exp(x * y)` using a 
    2nd-degree Maclaurin series polynomial: 1 + (xy) + (xy)^2/2.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
            - Column 0 corresponds to theta_0.
            - Column 1 corresponds to theta_1.
        return_flat (bool, optional): If True, flattens the output array before returning. 
            Defaults to True.

    Returns:
        np.ndarray: The evaluated low-fidelity approximation. Shape is 1D if 
        return_flat is True, otherwise maintains the broadcasted grid shape.
    """
    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    char_length = 2
    
    # Sample 100 points for x and y ranging from -1 to 1
    x = np.linspace(-1 * char_length, char_length, 100)
    y = np.linspace(-1 * char_length, char_length, 100)

    # Create a 2D meshgrid (100x100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the joint xy term once to optimize computation
    xy = X * Y  
    
    # Calculate the 3rd degree Maclaurin series approximation
    # Z = theta_0 * (1 + xy + xy^2/2! + xy^3/3!) + theta_1
    if return_flat:
        return theta_0 * (
            1 + 
            (xy) + 
            (xy**2 / 2)
        ).flatten() + theta_1
    else:
        return theta_0 * (
            1 + 
            (xy) + 
            (xy**2 / 2)
        ) + theta_1

def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, return_flat: bool = True) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the true model parameters.
        noise_st_dev (float): The standard deviation of the Gaussian noise to be added.
        return_flat (bool, optional): If True, flattens the output array before returning. 
            Defaults to True.

    Returns:
        np.ndarray: The simulated noisy data. Shape is 1D if return_flat is True, 
        otherwise maintains the High-Fidelity output shape.
    """
    # Run the "true" high-fidelity simulation
    Z_HF = M_HF(THETA)
    
    # Generate Gaussian (normal) noise with mean=0 and the specified standard deviation
    # The noise array matches the exact shape of the HF output
    random_noise = np.random.normal(0, noise_st_dev, size=Z_HF.shape)

    # Add noise to the true signal
    noisy_data = random_noise + Z_HF
    
    if return_flat:
        return noisy_data.flatten()
    else:
        return noisy_data