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

from exp_2d import M_HF, generate_noisy_data
from exp_2d import M_HF as M_LF
from smcpy.mfmc_proposal import MultiFidelityProposal
from smcpy.proposals import MultivarIndependent

from plotting_helpers import plot_2d_joint_posterior, plot_param_hists, plot_target_boxplots

# Data generation details
STD_DEV = 0.5
theta_0 = 1/20
theta_1 = 1
THETA_TRUE = np.array([[theta_0, theta_1]])
NUM_PARTICLES = 1_000
np.random.seed(42)
noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV)

'''
Execute MF SMC
'''

# Setup low-fidelity case
priors = [uniform(0.001, 2), uniform(0, 4)]
vector_mcmc = VectorMCMC(M_LF, noisy_data, priors, STD_DEV)

# initialize from prior
mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("theta_0", "theta_1"))
smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
lofi_step_list, lofi_mll_list = smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=5,
    target_ess=0.75
)
lofi_phi_list = smc.phi_sequence
lofi_particles = lofi_step_list[-1].params

# Setup low-fidelity posterior as proposal for high-fidelity
lofi_proposal_dist = MultiFidelityProposal(
    lofi_particles, 
    M_LF, 
    noisy_data,
    priors,
    STD_DEV
)
mcmc_kernel = VectorMCMCKernel(
    vector_mcmc, param_order=("a", "b"), path=GeometricPath(proposal=lofi_proposal_dist)
)

# Setup high-fidelity case
hifi_smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
hifi_step_list, hifi_mll_list = hifi_smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=5,
    target_ess=0.75,
)
hifi_phi_list = hifi_smc.phi_sequence


'''
Plot results
'''
run_label = 'plots/mfmc'
plot_target_boxplots(
    THETA_TRUE.flatten(),
    run_label,
    Low_Fidelity=(lofi_step_list, lofi_phi_list),
    High_Fidelity=(hifi_step_list, hifi_phi_list),
)

plot_2d_joint_posterior(
    THETA_TRUE.flatten(),
    run_label,
    Low_Fidelity=lofi_step_list,
    High_Fidelity=hifi_step_list,
)

plot_param_hists(
    THETA_TRUE.flatten(),
    run_label,
    Low_Fidelity=lofi_step_list,
    High_Fidelity=hifi_step_list
    )