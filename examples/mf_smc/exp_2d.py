import numpy as np
from typing import Union

def _generate_grid_standard(char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """
    Helper function to generate a 1D spatial grid clustered at the edges.
    """
    x = np.linspace(0, char_length, resolution)
    return x


def M_HF(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model over a 1D spatial domain.
    Z = theta_0 * exp(x) + theta_1
    """
    x = _generate_grid_standard(char_length, resolution)
    
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    return theta_0 * np.exp(x) + theta_1


def M_LF2(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF2 model using a 2nd-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2)
    ) + theta_1


def M_LF3(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF3 model using a 3rd-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2) + (x**3 / 6)
    ) + theta_1


def M_LF4(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF4 model using a 4th-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2) + (x**3 / 6) + (x**4 / 24)
    ) + theta_1


def M_LF5(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF5 model using a 5th-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2) + (x**3 / 6) + (x**4 / 24) +
        (x**5 / 120)
    ) + theta_1


def M_LF6(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF6 model using a 6th-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2) + (x**3 / 6) + (x**4 / 24) +
        (x**5 / 120) + (x**6 / 720)
    ) + theta_1


def M_LF7(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF7 model using a 7th-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2) + (x**3 / 6) + (x**4 / 24) +
        (x**5 / 120) + (x**6 / 720) + (x**7 / 5040)
    ) + theta_1


def M_LF8(THETA: np.ndarray, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF8 model using an 8th-degree Maclaurin series."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        1 + x + (x**2 / 2) + (x**3 / 6) + (x**4 / 24) +
        (x**5 / 120) + (x**6 / 720) + (x**7 / 5040) + (x**8 / 40320)
    ) + theta_1


def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, random_seed: int = None, char_length: float = 4.0, resolution: int = 10_000) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.
    """
    # Run the "true" high-fidelity simulation
    Z_HF = M_HF(THETA, char_length=char_length, resolution=resolution)
    
    # Generate Gaussian (normal) noise safely
    rng = np.random.default_rng(random_seed)
    random_noise = rng.normal(0, noise_st_dev, size=Z_HF.shape)

    # Add noise to the true signal
    noisy_data = Z_HF + random_noise
    
    return noisy_data