import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.stats import uniform, multivariate_normal, Normal, norm

sys.path.append(os.path.join(os.getcwd(), "../../"))
sys.path.append(os.path.join(os.getcwd(), "../../smcpy"))
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler as Sampler
from smcpy.paths import GeometricPath

from exp_3d import M_HF as M_HF
from exp_3d import generate_noisy_data
from smcpy.mfmc_proposal import MultiFidelityProposal
from smcpy.proposals import MultivarIndependent

from plotting_helpers import plot_2d_joint_posterior, plot_param_hists, plot_target_boxplots, plot_log_likelihood, save_run_hyperparameters

# Data generation details
STD_DEV = 5
theta_0 = 1/20
theta_1 = 1
THETA_TRUE = np.array([[theta_0, theta_1]])
NUM_PARTICLES = 5_000
random_seed = 16
noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV, random_seed=random_seed)

target_ess = 0.97
num_mcmc_samples = 7

'''
Execute MF SMC
'''

# Setup low-fidelity case
priors = [uniform(0.001, 2), uniform(0, 4)]
vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)

# initialize from prior
mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("theta_0", "theta_1"))
smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
hifi_step_list, hifi_mll_list = smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=num_mcmc_samples,
    target_ess=target_ess
)
hifi_phi_list = smc.phi_sequence
hifi_particles = hifi_step_list[-1].params

'''
Plot results
'''
run_label = 'plots/HIFI_REF'
np.save(run_label.split('/')[-1] + '_noisy_data.npy', noisy_data)
np.save(run_label.split('/')[-1] + '_posterior_particles.npy', hifi_particles)
plot_target_boxplots(
    THETA_TRUE.flatten(),
    run_label,
    High_Fidelity=(hifi_step_list, hifi_phi_list)
)

plot_2d_joint_posterior(
    THETA_TRUE.flatten(),
    run_label,
    High_Fidelity=hifi_step_list
)

plot_param_hists(
    THETA_TRUE.flatten(),
    run_label,
    High_Fidelity=hifi_step_list
)

plot_log_likelihood(
    run_label,
    High_Fidelity=hifi_mll_list
)

current_run_data = {
    "true_theta": THETA_TRUE.flatten().tolist() if hasattr(THETA_TRUE, 'tolist') else THETA_TRUE.flatten(),
    "target_ess": target_ess,
    "num_mcmc_samples": num_mcmc_samples,
    "num_particles": NUM_PARTICLES,
    "add_noise_stdev": STD_DEV,
    "random_seed": random_seed,
    "Extra details": 'Priors: [uniform(0.001, 2), uniform(0, 4)]\n Used exp_3d HF Model'
}

save_run_hyperparameters(
    'plots/run_info.json',
    run_label.split('/')[-1],
    **current_run_data
)

print(f"Run label: {run_label}")