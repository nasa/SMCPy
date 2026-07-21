import numpy as np
from typing import Union

def _generate_grid_standard(char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """
    Helper function to generate a 1D uniform spatial grid.
    """
    return np.linspace(-char_length, char_length, resolution)


def M_HF(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """
    Evaluates the High-Fidelity (HF) model over a 1D spatial domain.
    Z = theta_0 * sin(x) + theta_1
    """
    x = _generate_grid_standard(char_length, resolution)
    
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]

    return theta_0 * np.sin(x) + theta_1


def M_LF3(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF3 model using a 3rd-degree Maclaurin series for sine."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        x - (x**3 / 6)
    ) + theta_1


def M_LF5(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF5 model using a 5th-degree Maclaurin series for sine."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        x - (x**3 / 6) + (x**5 / 120)
    ) + theta_1


def M_LF7(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF7 model using a 7th-degree Maclaurin series for sine."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        x - (x**3 / 6) + (x**5 / 120) - (x**7 / 5040)
    ) + theta_1


def M_LF9(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF9 model using a 9th-degree Maclaurin series for sine."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        x - (x**3 / 6) + (x**5 / 120) - (x**7 / 5040) + 
        (x**9 / 362880)
    ) + theta_1


def M_LF11(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF11 model using an 11th-degree Maclaurin series for sine."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        x - (x**3 / 6) + (x**5 / 120) - (x**7 / 5040) + 
        (x**9 / 362880) - (x**11 / 39916800)
    ) + theta_1


def M_LF13(THETA: np.ndarray, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """Evaluates the LF13 model using a 13th-degree Maclaurin series for sine."""
    x = _generate_grid_standard(char_length, resolution)
    theta_0 = THETA[:, 0, None]
    theta_1 = THETA[:, 1, None]
    
    return theta_0 * (
        x - (x**3 / 6) + (x**5 / 120) - (x**7 / 5040) + 
        (x**9 / 362880) - (x**11 / 39916800) + (x**13 / 6227020800)
    ) + theta_1


def generate_noisy_data(THETA: np.ndarray, noise_st_dev: float, random_seed: int = None, char_length: float = 6.0, resolution: int = 10_000) -> np.ndarray:
    """
    Generates synthetic observation data by adding Gaussian noise to the High-Fidelity model.
    """
    Z_HF = M_HF(THETA, char_length=char_length, resolution=resolution)
    
    rng = np.random.default_rng(random_seed)
    random_noise = rng.normal(0, noise_st_dev, size=Z_HF.shape)

    noisy_data = Z_HF + random_noise
    
    return noisy_data