import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time

from scipy.stats import uniform

from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.paths import GeometricPath
from smcpy.utils.plotter import *


def eval_model(theta):
    time.sleep(0.1)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(eval_model, std_dev, plot=True, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + rng.normal(0, std_dev, y_true.shape)
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
    rng = np.random.default_rng(200)

    std_dev = 2
    noisy_data = generate_data(eval_model, std_dev, plot=False, rng=rng)

    # require phi=0.2 be included in adaptive sequence
    path = GeometricPath(required_phi=0.2)

    priors = [uniform(0.0, 6.0), uniform(0.0, 6.0)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, ("a", "b"), path=path, rng=rng)

    smc = AdaptiveSampler(mcmc_kernel)
    step_list, mll_list = smc.sample(
        num_particles=500, num_mcmc_samples=5, target_ess=0.7
    )
    print(f"phi_sequence={smc.phi_sequence}")
    print(f"fbf norm index={smc.req_phi_index}")
    print("marginal log likelihood = {}".format(mll_list[-1]))
    print("parameter means = {}".format(step_list[-1].compute_mean()))

    sns.pairplot(pd.DataFrame(step_list[-1].param_dict))
    sns.mpl.pyplot.savefig("pairwise.png")
