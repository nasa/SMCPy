import numpy as np
from scipy.stats import uniform

# --- smcpy Imports ---
from smcpy import AdaptiveSampler as Sampler
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy.paths import GeometricPath
from smcpy.proposals import MultiFidelityProposal

# --- Local Project Imports ---
from exp_3d import M_HF, generate_noisy_data, M_LF
from plotting_helpers import (
    plot_2d_joint_posterior, 
    plot_param_posteriors, 
    plot_target_boxplots, 
)

# =============================================================================
# 1. Configuration & Data Generation
# =============================================================================
STD_DEV = 5
theta_0 = 1 / 20
theta_1 = 1.0
THETA_TRUE = np.array([[theta_0, theta_1]])

NUM_PARTICLES = 5_000
target_ess = 0.8
num_mcmc_samples = 7
random_seed = 16

# Paths
run_label = 'results/'
os.makedirs(run_label, exist_ok=True)

# Generate synthetic measurement data
char_length = 2
noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV, random_seed=random_seed, char_length=char_length)

# Define priors for the Bayesian inference
priors = [uniform(0.001, 2), uniform(0, 4)]

# =============================================================================
# 2. Execute Low-Fidelity SMC
# =============================================================================
print("\n" + "=" * 50)
print("Stage 1: Executing Low-Fidelity SMC")
print("=" * 50 + "\n")

# Setup low-fidelity MCMC kernel
vector_mcmc_lofi = VectorMCMC(M_LF, noisy_data, priors, STD_DEV)
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
print("Stage 2: Executing High-Fidelity SMC")
print("=" * 50 + "\n")

# Setup low-fidelity posterior as the proposal distribution for high-fidelity
lofi_proposal_dist = MultiFidelityProposal(
    lofi_particles, 
    M_LF, 
    noisy_data,
    priors,
    STD_DEV
)

# Setup high-fidelity MCMC kernel with the geometric path proposal
vector_mcmc_hifi = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)
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

plot_param_posteriors(
    THETA_TRUE.flatten(),
    run_label,
    Low_Fidelity=lofi_step_list,
    High_Fidelity=hifi_step_list
)

print(f"Run fully completed. Results saved to: {run_label}")