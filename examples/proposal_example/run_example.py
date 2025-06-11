import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.stats import uniform, multivariate_normal

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import FixedSampler as Sampler
from smcpy.paths import GeometricPath

sys.path.append(os.path.join(os.path.split(__file__)[0], "../"))
from helper_functions import eval_model, generate_data, plot_noisy_data


STD_DEV = 2
X_TRUE = np.array([[2, 3.5]])
NUM_PARTICLES = 500


def plot_target_means(**kwargs):
    _, axes = plt.subplots(2)

    for key, targets in kwargs.items():
        means = [target.compute_mean() for target in targets]
        axes[0].plot([m["a"] for m in means], label=key)
        axes[1].plot([m["b"] for m in means])

    axes[0].legend()
    axes[0].set_ylabel("$\mu_a$")
    axes[1].set_ylabel("$\mu_b$")
    axes[1].set_xlabel("target index")

    plt.show()


if __name__ == "__main__":
    np.random.seed(200)
    noisy_data = generate_data(X_TRUE, eval_model, STD_DEV, plot=False)

    priors = [uniform(-5, 10), uniform(-10, 20)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, STD_DEV)
    phi_sequence = np.linspace(0, 1, 20)

    # initialize from prior
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("a", "b"))
    smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
    reg_step_list, mll_list = smc.sample(
        num_particles=NUM_PARTICLES,
        num_mcmc_samples=5,
        phi_sequence=phi_sequence,
        ess_threshold=1.0,
    )

    # initialize from proposal
    proposal_dist = multivariate_normal(mean=np.array([2, 3]), cov=np.eye(2))
    mcmc_kernel = VectorMCMCKernel(
        vector_mcmc, param_order=("a", "b"), path=GeometricPath(proposal=proposal_dist)
    )
    smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
    prop_step_list, mll_list = smc.sample(
        num_particles=NUM_PARTICLES,
        num_mcmc_samples=5,
        phi_sequence=phi_sequence,
        ess_threshold=1.0,
    )

    plot_target_means(Prior=reg_step_list, Proposal=prop_step_list)
