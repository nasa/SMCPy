import numpy as np
from typing import Union

resolution = 10_000
char_length = 4

# Generate 1D uniform grid, range: [0, 4]
x = np.linspace(-char_length, char_length, resolution)

def M_HF(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model over a 1D spatial domain.
    
    The model evaluates the function: Z = theta_0 * sin(x) + theta_1
    over a 1D array of 10,000 points ranging from 0 to 4.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters. 
            - Column 0 corresponds to theta_0 (the scaling factor).
            - Column 1 corresponds to theta_1 (the bias/offset).

    Returns:
        np.ndarray: The evaluated high-fidelity output with shape (N, resolution).
    """
    # Extract parameters, keeping the first dimension to allow batch processing of N samples
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Evaluate the exact sine function
    output = theta_0 * np.sin(x) + theta_1
    
    return output

def M_LF(THETA: np.ndarray) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF) model using a Maclaurin series approximation.
    
    This approximates the high-fidelity sine term `sin(x)` using a 
    9th-degree Maclaurin series polynomial.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
            - Column 0 corresponds to theta_0.
            - Column 1 corresponds to theta_1.

    Returns:
        np.ndarray: The evaluated low-fidelity approximation with shape (N, resolution).
    """
    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    # Calculate the 9th degree Maclaurin series approximation for sine
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
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the model parameters.
            - Column 0 corresponds to theta_0.
            - Column 1 corresponds to theta_1.

    Returns:
        np.ndarray: The evaluated 3rd-degree low-fidelity approximation with shape (N, resolution).
    """
    # Extract parameters for batch processing
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    # Calculate the 3rd degree Maclaurin series approximation for sine
    # Z = theta_0 * (x - x^3/3!) + theta_1
    output = theta_0 * (
        x - 
        (x**3 / 6)
    ) + theta_1
    
    return output

def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.

    Args:
        THETA (np.ndarray): A 2D array of shape (N, 2) containing the true model parameters.
        noise_st_dev (float): The standard deviation of the Gaussian noise to be added.

    Returns:
        np.ndarray: The simulated noisy data with shape (N, resolution).
    """
    # Run the "true" high-fidelity simulation
    Z_HF = M_HF(THETA)
    
    # Generate Gaussian (normal) noise with mean=0 and the specified standard deviation
    random_noise = np.random.normal(0, noise_st_dev, size=Z_HF.shape)

    # Add noise to the true signal
    noisy_data = Z_HF + random_noise
    
    return noisy_data