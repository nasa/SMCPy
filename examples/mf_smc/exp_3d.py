import numpy as np
from typing import Union

def _generate_meshgrid_standard(char_length: int, resolution: int = 150):
    """Helper function to generate the 2D spatial meshgrid based on char_length."""
    x = np.linspace(np.pi, 0, resolution)
    y = np.linspace(np.pi, 0, resolution)

    x = char_length * np.cos(x)
    y = char_length * np.cos(y)

    return np.meshgrid(x, y)

def _generate_meshgrid_elliptical(char_length: float, resolution: int = 150, oval_width: float = 4.0, oval_height: float = 1.5, rotation_deg: float = 45.0):
    """
    Helper function to generate a 2D spatial meshgrid mapped to a rotated elliptical domain.
    
    Args:
        char_length (float): Used to generate the base Chebyshev grid.
        resolution (int): Number of points along each axis.
        oval_width (float): The x-axis radius of the ellipse.
        oval_height (float): The y-axis radius of the ellipse.
        rotation_deg (float): Degrees to rotate the final ellipse.
        
    Returns:
        tuple: (U_rotated, V_rotated) representing the 2D spatial grid.
    """
    # Generate the base Chebyshev grid
    x = np.linspace(np.pi, 0, resolution)
    y = np.linspace(np.pi, 0, resolution)

    x = char_length * np.cos(x)
    y = char_length * np.cos(y)

    X, Y = np.meshgrid(x, y)

    # 1. Normalize the grid to [-1, 1] for the mapping math to work
    X_norm = X / char_length
    Y_norm = Y / char_length

    oval_width = char_length

    # 2. Apply Elliptical Mapping (maps the square to a unit circle)
    U_norm = X_norm * np.sqrt(1 - (Y_norm**2) / 2.0)
    V_norm = Y_norm * np.sqrt(1 - (X_norm**2) / 2.0)

    # 3. Scale to an Oval / Ellipse
    U = U_norm * oval_width
    V = V_norm * oval_height

    # 4. Rotate the oval
    theta = np.radians(rotation_deg) 
    U_rotated = U * np.cos(theta) - V * np.sin(theta)
    V_rotated = U * np.sin(theta) + V * np.cos(theta)

    return U_rotated, V_rotated

_generate_meshgrid = _generate_meshgrid_standard

def M_HF(THETA: np.ndarray, return_flat: bool = True, char_length: int = 4) -> np.ndarray:
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

    if return_flat:
        return theta_0 * np.exp(X * Y).flatten() + theta_1
    else:
        return theta_0 * np.exp(X * Y) + theta_1


def M_LF2(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF2) model using a 2nd-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2)
    ).flatten() + theta_1


def M_LF3(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF3) model using a 3rd-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6)
    ).flatten() + theta_1


def M_LF4(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF4) model using a 4th-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6) + (xy**4 / 24)
    ).flatten() + theta_1


def M_LF5(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF5) model using a 5th-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6) + (xy**4 / 24) +
        (xy**5 / 120)
    ).flatten() + theta_1


def M_LF6(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF6) model using a 6th-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6) + (xy**4 / 24) +
        (xy**5 / 120) + (xy**6 / 720)
    ).flatten() + theta_1


def M_LF7(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF7) model using a 7th-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6) + (xy**4 / 24) +
        (xy**5 / 120) + (xy**6 / 720) + (xy**7 / 5040)
    ).flatten() + theta_1


def M_LF8(THETA: np.ndarray, char_length: int = 2) -> np.ndarray:
    """
    Evaluates the Low-Fidelity (LF8) model using an 8th-degree Maclaurin series.
    """
    X, Y = _generate_meshgrid(char_length)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    xy = X * Y  
    
    return theta_0 * (
        1 + (xy) + (xy**2 / 2) + (xy**3 / 6) + (xy**4 / 24) +
        (xy**5 / 120) + (xy**6 / 720) + (xy**7 / 5040) + (xy**8 / 40320)
    ).flatten() + theta_1


def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, return_flat: bool = True, random_seed: int = None, char_length: int = 4) -> np.ndarray:
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
    Z_HF = M_HF(THETA, char_length=char_length, return_flat=False)
    
    # Generate reproducible noise
    rng = np.random.default_rng(random_seed)
    random_noise = rng.normal(0, noise_st_dev, size=Z_HF.shape)

    # Add noise to the true signal
    noisy_data = random_noise + Z_HF
    
    if return_flat:
        return noisy_data.flatten()
    else:
        return noisy_data