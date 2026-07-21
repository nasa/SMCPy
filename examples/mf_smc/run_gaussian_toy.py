import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.append(os.path.join(os.getcwd(), "../../"))
sys.path.append(os.path.join(os.getcwd(), "../../smcpy"))
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy.smc.samplers import AdaptiveSampler
from smcpy.paths import GeometricPath
from SMCPy.smcpy.mfsmc_proposal import MultiFidelityProposal

# Import our new toy problem functions
from gaussian_toy import generate_noisy_data, M_LF, M_HF, get_analytical_posterior

# ==========================================
# 1. Setup Parameters
# ==========================================
THETA_TRUE = 5.0
NOISE_STD = 1.0

# Define fidelity by sample size
N_LF = 5     # Low fidelity: only 5 noisy observations
N_HF = 500   # High fidelity: 500 observations

NUM_PARTICLES = 2000

# Prior definition: Normal(mean=0, std=10) - a relatively weak/flat prior
PRIOR_MEAN = 0.0
PRIOR_STD = 10.0
priors = [norm(loc=PRIOR_MEAN, scale=PRIOR_STD)]

# ==========================================
# 2. Generate Data
# ==========================================
# Generate the massive high-fidelity dataset
hf_data = generate_noisy_data(THETA_TRUE, NOISE_STD, N_HF, random_seed=42)

# The low-fidelity data is just the first N_LF points from the same sequence
lf_data = hf_data[:N_LF]

# Calculate analytical posteriors for verification
lf_true_mean, lf_true_std = get_analytical_posterior(lf_data, NOISE_STD, PRIOR_MEAN, PRIOR_STD)
hf_true_mean, hf_true_std = get_analytical_posterior(hf_data, NOISE_STD, PRIOR_MEAN, PRIOR_STD)

print(f"--- Analytical Truths ---")
print(f"True Theta:        {THETA_TRUE}")
print(f"LF Posterior:      Mean = {lf_true_mean:.4f}, Std = {lf_true_std:.4f}")
print(f"HF Posterior:      Mean = {hf_true_mean:.4f}, Std = {hf_true_std:.4f}")
print("-" * 25)

# ==========================================
# 3. Run Low-Fidelity SMC
# ==========================================
print("\nRunning Low-Fidelity SMC...")
def lf_model_wrapper(theta):
    return M_LF(theta, N_LF)

lf_vector_mcmc = VectorMCMC(lf_model_wrapper, lf_data, priors, NOISE_STD)
lf_kernel = VectorMCMCKernel(lf_vector_mcmc, param_order=("theta",))
lf_smc = AdaptiveSampler(mcmc_kernel=lf_kernel, show_progress_bar=True)

lf_steps, lf_mll = lf_smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=10,
    target_ess=0.8
)

lf_final_particles = lf_steps[-1].params[:, 0]
print(f"LF SMC Result: Mean = {np.mean(lf_final_particles):.4f}, Std = {np.std(lf_final_particles):.4f}")

# ==========================================
# 4. Construct Multi-Fidelity Proposal
# ==========================================
print("\nConstructing Multi-Fidelity Proposal from LF Posterior...")
# Extract samples from the final step of the LF run
lofi_samples = lf_steps[-1].params

# Pass ALL required parameters matching the original script's signature
mf_proposal = MultiFidelityProposal(
    lofi_samples, 
    lf_model_wrapper, 
    lf_data,
    priors,
    NOISE_STD
)

# ==========================================
# 5. Run High-Fidelity SMC using MF Proposal
# ==========================================
print("\nRunning High-Fidelity SMC using Multi-Fidelity Proposal...")
def hf_model_wrapper(theta):
    return M_HF(theta, N_HF)

hf_vector_mcmc = VectorMCMC(hf_model_wrapper, hf_data, priors, NOISE_STD)

# Initialize the kernel using the proposal path directly with the mf_proposal object
hf_kernel = VectorMCMCKernel(
    hf_vector_mcmc, 
    param_order=("theta",), 
    path=GeometricPath(proposal=mf_proposal)
)

mf_smc = AdaptiveSampler(mcmc_kernel=hf_kernel, show_progress_bar=True)

mf_steps, mf_mll = mf_smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=10,
    target_ess=0.8
)

mf_final_particles = mf_steps[-1].params[:, 0]
print(f"HF SMC Result: Mean = {np.mean(mf_final_particles):.4f}, Std = {np.std(mf_final_particles):.4f}")

# ==========================================
# 6. Plot Results
# ==========================================
plt.figure(figsize=(10, 6))

# Plot analytical truths
x_axis = np.linspace(3.5, 6.5, 1000)
plt.plot(x_axis, norm.pdf(x_axis, loc=lf_true_mean, scale=lf_true_std),
         'b--', label='Analytical LF Posterior', linewidth=2)
plt.plot(x_axis, norm.pdf(x_axis, loc=hf_true_mean, scale=hf_true_std),
         'r-', label='Analytical HF Posterior', linewidth=2)

# Plot SMC Results
plt.hist(lf_final_particles, bins=50, density=True, alpha=0.5, color='blue', label='LF SMC Particles')
plt.hist(mf_final_particles, bins=50, density=True, alpha=0.5, color='red', label='MF SMC Particles (Final)')

plt.axvline(THETA_TRUE, color='black', linestyle=':', label='True Theta', linewidth=2)

plt.title('Multi-Fidelity SMC: 1D Gaussian Toy Problem')
plt.xlabel('Theta')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('gaussian_toy_results.png')
print("\nPlot saved to 'gaussian_toy_results.png'")
plt.show()