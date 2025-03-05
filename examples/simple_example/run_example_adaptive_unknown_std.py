import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import uniform

from smcpy import AdaptiveSampler, VectorMCMC, VectorMCMCKernel

from run_example_adaptive import generate_data, eval_model

if __name__ == "__main__":
    rng = np.random.default_rng(seed=200)

    noisy_data = generate_data(eval_model, std_dev=2, plot=False, rng=rng)

    std_dev = None  # estimate it; requires prior for std_dev and param_order

    priors = [uniform(0.0, 6.0), uniform(0.0, 6.0), uniform(0, 10)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("a", "b", "std"), rng=rng)

    smc = AdaptiveSampler(mcmc_kernel)
    step_list, mll_list = smc.sample(
        num_particles=500, num_mcmc_samples=5, target_ess=0.8
    )

    print("marginal log likelihood = {}".format(mll_list[-1]))
    print("parameter means = {}".format(step_list[-1].compute_mean()))

    sns.pairplot(pd.DataFrame(step_list[-1].param_dict))
    sns.mpl.pyplot.savefig("pairwise.png")
