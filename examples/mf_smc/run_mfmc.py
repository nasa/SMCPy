import os
import sys
import numpy as np

# --- System Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../smcpy")))

from scipy.stats import uniform

# --- smcpy Imports ---
from smcpy import AdaptiveSampler as Sampler
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy.paths import GeometricPath
from smcpy.mfmc_proposal import MultiFidelityProposal

# --- Local Project Imports ---
from exp_3d import M_HF, generate_noisy_data
from exp_3d import M_LF5 as M_LF
from plotting_helpers import (
    plot_2d_joint_posterior, 
    plot_param_hists, 
    plot_target_boxplots, 
    save_run_hyperparameters
)

# =============================================================================
# 1. Configuration & Data Generation
# =============================================================================
STD_DEV = 5
theta_0 = 1 / 20
theta_1 = 1.0
THETA_TRUE = np.array([[theta_0, theta_1]])

NUM_PARTICLES = 5_000
target_ess = 0.5
num_mcmc_samples = 7
random_seed = 16

# Paths
run_label = '/hpnobackup2/vasanche/exp_3d_case/varying_fidelities/test'
log_filename = '/hpnobackup2/vasanche/exp_3d_case/varying_fidelities/run_info.json'

# Generate synthetic measurement data
lofi_char_length = 2
hifi_char_length = 4
lofi_noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV, random_seed=random_seed, char_length=lofi_char_length)
hifi_noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV, random_seed=random_seed, char_length=hifi_char_length)
# lofi_noisy_data = hifi_noisy_data
# all(hifi_noisy_data == lofi_noisy_data)

def wrapper_M_HF(THETA):
    return M_HF(THETA, char_length=lofi_char_length)
def wrapper_M_LF(THETA):
    return M_LF(THETA, char_length=hifi_char_length)

# Define priors for the Bayesian inference
priors = [uniform(0.001, 2), uniform(0, 4)]

# =============================================================================
# 2. Execute Low-Fidelity SMC
# =============================================================================
print("\n" + "=" * 50)
print("Phase 1: Executing Low-Fidelity SMC")
print("=" * 50 + "\n")

# Setup low-fidelity MCMC kernel
vector_mcmc_lofi = VectorMCMC(wrapper_M_LF, lofi_noisy_data, priors, STD_DEV)
mcmc_kernel_lofi = VectorMCMCKernel(vector_mcmc_lofi, param_order=("theta_0", "theta_1"))

# Run SMC
lofi_smc = Sampler(mcmc_kernel=mcmc_kernel_lofi, show_progress_bar=True)
lofi_step_list, lofi_mll_list = lofi_smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=num_mcmc_samples,
    target_ess=target_ess
)

# Extract results
lofi_phi_list = lofi_smc.phi_sequence
lofi_particles = lofi_step_list[-1].params

# =============================================================================
# 3. Execute High-Fidelity SMC
# =============================================================================
print("\n" + "=" * 50)
print("Phase 2: Executing High-Fidelity SMC")
print("=" * 50 + "\n")

# Setup low-fidelity posterior as the proposal distribution for high-fidelity
lofi_proposal_dist = MultiFidelityProposal(
    lofi_particles, 
    wrapper_M_LF, 
    lofi_noisy_data,
    priors,
    STD_DEV
)

# Setup high-fidelity MCMC kernel with the geometric path proposal
vector_mcmc_hifi = VectorMCMC(wrapper_M_HF, hifi_noisy_data, priors, STD_DEV)
mcmc_kernel_hifi = VectorMCMCKernel(
    vector_mcmc_hifi, 
    param_order=("theta_0", "theta_1"), 
    path=GeometricPath(proposal=lofi_proposal_dist)
)

# Run SMC
hifi_smc = Sampler(mcmc_kernel=mcmc_kernel_hifi, show_progress_bar=True)
hifi_step_list, hifi_mll_list = hifi_smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=num_mcmc_samples,
    target_ess=target_ess,
)

# Extract results
hifi_phi_list = hifi_smc.phi_sequence

# =============================================================================
# 4. Plotting Results
# =============================================================================
print("\n" + "=" * 50)
print("Generating and Saving Plots...")
print("=" * 50)

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

# =============================================================================
# 5. Log Run Information to JSON
# =============================================================================
print("Saving run hyperparameters to log file...")

current_run_data = {
    "true_theta": THETA_TRUE.flatten().tolist(),
    "target_ess": target_ess,
    "num_mcmc_samples": num_mcmc_samples,
    "num_particles": NUM_PARTICLES,
    "add_noise_stdev": STD_DEV,
    "random_seed": random_seed,
    "lofi_num_smc_steps": len(lofi_phi_list),
    "hifi_num_smc_steps": len(hifi_phi_list),
    "Extra details": "Priors: [uniform(0.001, 2), uniform(0, 4)]\n Used exp_3d HF Model & LF8 Model"
}

save_run_hyperparameters(
    log_filename,
    run_label,
    **current_run_data
)

print(f"Run fully completed. Results saved to: {run_label}")