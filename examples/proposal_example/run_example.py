import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.stats import uniform, multivariate_normal

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler as Sampler
from smcpy.paths import GeometricPath

sys.path.append(os.path.join(os.path.split(__file__)[0], "../"))
from helper_functions import eval_model, generate_data, plot_noisy_data

STD_DEV = 2
X_TRUE = np.array([[2, 3.5]])
NUM_PARTICLES = 1000


def plot_target_boxplots(true_values, **series):
    """Plot box plots for each parameter and series over the phi sequence.

    Args:
        true_values: array-like of true parameter values in param order
        **series: label=(targets_list, phi_sequence) for each SMC run
    """
    first_targets, _ = next(iter(series.values()))
    param_names = first_targets[0].param_names
    n_params = len(param_names)
    n_series = len(series)

    fig, axes = plt.subplots(
        n_params,
        n_series,
        sharex="col",
        sharey="row",
        figsize=(4 * n_series, 3 * n_params),
    )
    axes = np.atleast_2d(axes)

    for col, (label, (targets, phi_sequence)) in enumerate(series.items()):
        positions = np.arange(len(phi_sequence))
        box_width = 0.6
        for row, (name, true_val) in enumerate(zip(param_names, true_values)):
            ax = axes[row, col]
            ax.boxplot(
                [target.params[:, row] for target in targets],
                positions=positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
            )
            ax.axhline(
                true_val,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label="true value",
                alpha=0.7,
            )
            ax.grid(True)
            if col == 0:
                ax.set_ylabel(f"${name}$")
        axes[0, col].set_title(label + f" ({len(targets)} steps)")

    for col in range(n_series):
        axes[-1, col].set_xlabel("step")

    axes[0, -1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("target_boxplots.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(200)
    noisy_data = generate_data(X_TRUE, eval_model, STD_DEV, plot=False)

    priors = [uniform(-10, 20), uniform(-10, 20)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, STD_DEV)
    # phi_sequence = np.linspace(0, 1, 20)

    # initialize from prior
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("a", "b"))
    smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
    reg_step_list, mll_list = smc.sample(
        num_particles=NUM_PARTICLES,
        num_mcmc_samples=5,
        target_ess=0.5,
        # phi_sequence=phi_sequence,
        # ess_threshold=1.0,
    )
    reg_phi_list = smc.phi_sequence

    # initialize from proposal
    proposal_dist = multivariate_normal(mean=np.array([2, 3]), cov=np.eye(2))
    mcmc_kernel = VectorMCMCKernel(
        vector_mcmc, param_order=("a", "b"), path=GeometricPath(proposal=proposal_dist)
    )
    smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
    prop_step_list, mll_list = smc.sample(
        num_particles=NUM_PARTICLES,
        num_mcmc_samples=5,
        target_ess=0.5,
        # phi_sequence=phi_sequence,
        # ess_threshold=1.0,
    )
    prop_phi_list = smc.phi_sequence

    plot_target_boxplots(
        X_TRUE.flatten(),
        prior=(reg_step_list, reg_phi_list),
        proposal=(prop_step_list, prop_phi_list),
    )
