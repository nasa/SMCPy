import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import FixedPhiSampler, AdaptiveSampler
from smcpy.utils.plotter import *
from smcpy.utils.storage import HDF5Storage


def eval_model(theta):
    time.sleep(0.1)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), "-k")
    ax.plot(x.flatten(), noisy_data.flatten(), "o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    np.random.seed(200)

    filename = "example.h5"

    num_smc_steps = 5
    std_dev = 2
    noisy_data = generate_data(eval_model, std_dev, plot=False)

    priors = [uniform(0.0, 6.0), uniform(0.0, 6.0)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("a", "b"))

    # Run SMC with an incomplete phi sequence (terminates before phi = 1)
    phi_seq = np.linspace(0, 0.25, num_smc_steps)
    with HDF5Storage(filename, mode="w"):
        smc = FixedPhiSampler(mcmc_kernel)
        results, mll = smc.sample(
            num_particles=500,
            num_mcmc_samples=5,
            ess_threshold=0.7,
            phi_sequence=phi_seq,
        )

    # Restart SMC run (can even use a different sampler if desired)
    with HDF5Storage(filename, mode="a"):
        smc = AdaptiveSampler(mcmc_kernel)
        results, mll = smc.sample(num_particles=500, num_mcmc_samples=5, target_ess=0.8)

    print("marginal log likelihood = {}".format(mll[-1]))
    print("parameter means = {}".format(results[-1].compute_mean()))
    plot_pairwise(results[-1].params, results[-1].weights, param_names=["a", "b"])
