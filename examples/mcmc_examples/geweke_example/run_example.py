import numpy as np

from scipy.signal import welch

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy import ImproperUniform
from smcpy.utils.plotter import *
from smcpy.utils.geweke import compute_geweke


if __name__ == "__main__":
    np.random.seed(2)

    true_mean = 1
    true_std = 2
    n_data_pts = 100
    n_samples = 6000
    burnin = 0
    param_names = ["mean", "std"]

    model = lambda x: np.tile(x[:, 0], (1, n_data_pts))
    data = np.random.normal(1, 2, n_data_pts)
    priors = [ImproperUniform(), ImproperUniform(0, None)]
    x0 = np.array([[-3, 7]])
    cov = np.eye(2) * 0.001

    vmcmc = VectorMCMC(model, data, priors)
    chain = vmcmc.metropolis(
        x0, n_samples, cov, adapt_interval=100, adapt_delay=1000, progress_bar=True
    )

    burnin, z = compute_geweke(chain[0], window_pct=10, step_pct=1)

    plot_mcmc_chain(chain, param_names)
    plot_geweke(burnin, z, param_names)
