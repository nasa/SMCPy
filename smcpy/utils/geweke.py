import numpy as np

from scipy.signal import welch


def compute_geweke(x, window_pct=10, step_pct=1):
    """
    :param x: MCMC chain with shape (number samples, number parameters)
    :type x: 2D array
    :param window_pct: percentage of total samples to use when computing
        the Geweke score
    :type window_pct: int or float
    :param step_pct: window shift for each Geweke score calculation given
        as a percentage of the total samples
    :type step_pct: int or float
    """
    if window_pct > 50 or window_pct <= 0:
        raise ValueError("window size must be in [0, 50] percent")

    n_samples = x.shape[1]

    burnins_to_test = np.arange(0, 50 - window_pct + step_pct, step_pct)
    burnins_to_test = burnins_to_test * n_samples / 100

    window = int(window_pct * n_samples / 100)

    x2 = x[:, int(0.5 * n_samples) :]
    psd2 = _spec_density(x2)
    mean2 = x2.mean(axis=1)

    z = []
    burnin = []
    for burn in burnins_to_test:
        burnin.append(int(burn))

        x1 = x[:, burnin[-1] : burnin[-1] + window]
        psd1 = _spec_density(x1)
        mean1 = x1.mean(axis=1)

        z.append((mean1 - mean2) / np.sqrt(psd1 / x1.shape[1] + psd2 / x2.shape[1]))

    return burnin, np.array(z)


def _spec_density(x):
    psd = welch(x, detrend=False, axis=1, scaling="spectrum")[1][:, 0]
    return psd
