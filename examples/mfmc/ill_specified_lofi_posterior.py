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

from exp_3d import M_HF, generate_noisy_data
from smcpy.mfmc_proposal import MultiFidelityProposal
from smcpy.proposals import MultivarIndependent

from plotting_helpers import plot_ill_posed_res
import datetime

# Data generation details
STD_DEV = 0.5
theta_0 = 1/20
theta_1 = 1
THETA_TRUE = np.array([[theta_0, theta_1]])
NUM_PARTICLES = 1_000
np.random.seed(42)
noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV)

def perturb_lofi_posterior(particles, bias_adjust, stdev_adjust):
    
    # adjust the variance
    perturbed_particles = (particles - np.mean(particles, axis = 0)) * stdev_adjust + np.mean(particles, axis=0)  

    # adjust the bias
    perturbed_particles = perturbed_particles + bias_adjust
    
    return perturbed_particles

def mfmc_for_hifi_stage(perturbed_lofi_particles, priors, noisy_data, std_dev):
    # Setup low-fidelity posterior as proposal for high-fidelity
    lofi_proposal_dist = MultiFidelityProposal(
        perturbed_lofi_particles, 
        M_HF, 
        noisy_data,
        priors,
        std_dev
    )
    
    # Create the high-fidelity MCMC definition
    hifi_vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, std_dev)
    
    mcmc_kernel = VectorMCMCKernel(
        hifi_vector_mcmc, 
        param_order=("theta_0", "theta_1"), 
        path=GeometricPath(proposal=lofi_proposal_dist)
    )

    # Setup high-fidelity case
    hifi_smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
    hifi_step_list, hifi_mll_list = hifi_smc.sample(
        num_particles=NUM_PARTICLES,
        num_mcmc_samples=5,
        target_ess=0.9,
    )
    hifi_phi_list = hifi_smc.phi_sequence

    return hifi_step_list, hifi_phi_list

'''
Execute MF SMC
'''

# Setup low-fidelity case
priors = [uniform(0.001, 2), uniform(0, 4)]
# vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)

# # initialize from prior
# mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("theta_0", "theta_1"))
# smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
# lofi_step_list, lofi_mll_list = smc.sample(
#     num_particles=NUM_PARTICLES,
#     num_mcmc_samples=5,
#     target_ess=0.75
# )
# lofi_phi_list = smc.phi_sequence
# lofi_particles = lofi_step_list[-1].params
# np.save('pseudo_lofi_posterior.npy', lofi_particles)

# read in pseudo lofi posterior particles
lofi_particles = np.load('pseudo_lofi_posterior.npy')


bias_adjustments_arr = np.array([-0.001,-.01])
stdev_adjustments_arr = np.array([1,1])

# bias_adjustments_arr = np.array([0,0])
# stdev_adjustments_arr = np.array([1,1])

perturbed_particles = perturb_lofi_posterior(lofi_particles, bias_adjustments_arr, stdev_adjustments_arr)
hifi_step_list, hifi_phi_list = mfmc_for_hifi_stage(perturbed_particles, priors, noisy_data, STD_DEV)

'''
Plot results
'''
run_label = 'figs_ill_posed/bias_negative'
plot_ill_posed_res(
    THETA_TRUE.flatten(),
    run_label,
    perturbed_particles,
    bias_adjustments_arr, stdev_adjustments_arr,
    High_Fidelity=(hifi_step_list, hifi_phi_list)
)

'''
Log Run Information to TXT File
'''
# Define the name of your master log file
log_filename = "figs_ill_posed/run_info.txt"

# Get the current time for the timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Format the text exactly how you want it to appear in the file
log_entry = (
    f"[{current_time}] Run Label: {run_label}\n"
    f"    Bias Adjustments (theta_0, theta_1):     {bias_adjustments_arr}\n"
    f"    St.Dev. Adjustments (theta_0, theta_1): {stdev_adjustments_arr}\n"
    f"    True Theta:                              {THETA_TRUE.flatten()}\n"
    f"------------------------------------------------------------------\n"
)

# Open the file in append mode ('a') so it adds to the bottom instead of overwriting
with open(log_filename, "a") as log_file:
    log_file.write(log_entry)

print(f"Run details successfully logged to {log_filename}")