import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import uniform

from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.utils.noise_generator import generate_noisy_data


def model(params):
    return params[:, [0]] * np.linspace(0.5, 2.5, 100) + params[:, [1]]


if __name__ == "__main__":

    std_dev = 0.5
    true_params = np.array([[2, 3.5]])
    noisy_data = generate_noisy_data(model(true_params), std_dev)
    priors = [uniform(-6, 12.0), uniform(-6, 12.0)]

    vector_mcmc = VectorMCMC(model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, ("slope", "intercept"))

    smc = AdaptiveSampler(mcmc_kernel)
    step_list, mll_list = smc.sample(num_particles=500, num_mcmc_samples=10)

    print(f"marginal log likelihood = {mll_list[-1]}")
    print(f"parameter means = {step_list[-1].compute_mean()}")

    sns.pairplot(pd.DataFrame(step_list[-1].param_dict))
    sns.mpl.pyplot.savefig("pairwise.png")
