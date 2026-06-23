import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json

from scipy.stats import uniform, wasserstein_distance

sys.path.append(os.path.join(os.getcwd(), "../../"))
sys.path.append(os.path.join(os.getcwd(), "../../smcpy"))
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler as Sampler
from smcpy.paths import GeometricPath

from exp_3d import M_HF
from smcpy.mfmc_proposal import MultiFidelityProposal
from smcpy.proposals import MultivarIndependent

from plotting_helpers import plot_ill_posed_res, save_run_hyperparameters, plot_param_hist_progression, plot_target_boxplots
import datetime

##########################################################################################
##########################################################################################
##########################################################################################
'''
Run details
'''
# Data generation details
STD_DEV = 5
theta_0 = 1/20
theta_1 = 1
THETA_TRUE = np.array([[theta_0, theta_1]])
NUM_PARTICLES = 5_000
noisy_data = np.load('HIFI_REF_noisy_data.npy')

run_label = 'figs_ill_posed/theta1_bias5'
bias_adjustments_arr = np.array([0., -0.1])
stdev_adjustments_arr = np.array([1,1])
target_ess = 0.99
num_mcmc_samples = 10
##########################################################################################
##########################################################################################
##########################################################################################

def perturb_lofi_posterior(particles, bias_adjust, stdev_adjust):
    
    # adjust the variance
    perturbed_particles = (particles - np.mean(particles, axis = 0)) * stdev_adjust + np.mean(particles, axis=0)  

    # adjust the bias
    perturbed_particles = perturbed_particles + bias_adjust
    
    return perturbed_particles

'''
Execute MF SMC
'''
lofi_particles = np.load('HIFI_REF_posterior_particles.npy')

perturbed_particles = perturb_lofi_posterior(lofi_particles, bias_adjustments_arr, stdev_adjustments_arr)

# Setup low-fidelity posterior as proposal for high-fidelity
priors = [uniform(0.001, 2), uniform(0, 4)]
lofi_proposal_dist = MultiFidelityProposal(
    perturbed_particles, 
    M_HF, 
    noisy_data,
    priors,
    STD_DEV
)

# Create the high-fidelity MCMC definition
hifi_vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)

mcmc_kernel = VectorMCMCKernel(
    hifi_vector_mcmc, 
    param_order=("theta_0", "theta_1"), 
    path=GeometricPath(proposal=lofi_proposal_dist)
)

# Setup high-fidelity case
hifi_smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
hifi_step_list, hifi_mll_list = hifi_smc.sample(
    num_particles=NUM_PARTICLES,
    num_mcmc_samples=num_mcmc_samples,
    target_ess=target_ess,
)
hifi_phi_list = hifi_smc.phi_sequence
hifi_particles = hifi_step_list[-1].params

'''
Plot results
'''
plot_ill_posed_res(
    THETA_TRUE.flatten(),
    run_label,
    perturbed_particles,
    bias_adjustments_arr, stdev_adjustments_arr,
    High_Fidelity=(hifi_step_list, hifi_phi_list)
)

plot_param_hist_progression(
    run_label,
    lofi_particles,
    perturbed_particles,
    hifi_particles,
    bias_adjustments_arr, stdev_adjustments_arr
)

plot_target_boxplots(
    THETA_TRUE.flatten(),
    run_label,
    High_Fidelity = (hifi_step_list, hifi_phi_list)
)

'''
Wasserstein distance and posterior mean difference
'''
wasser_dist = []
n_params = lofi_particles.shape[-1]

for i in range(n_params):
    dist = wasserstein_distance(lofi_particles[:, i], hifi_particles[:, i])
    wasser_dist.append(dist)

hifi_post_mean = np.mean(hifi_particles, axis = 0)
lofi_post_mean = np.mean(lofi_particles, axis = 0)

post_mean_difference = hifi_post_mean - lofi_post_mean

'''
Log Run Information to JSON File
'''
# Define the name of your master log file
log_filename = "figs_ill_posed/run_info.json"

# Get the current time for the timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 1. Package the current run's data into a dictionary
# (We use .tolist() just in case these are NumPy arrays, which JSON cannot serialize directly)
current_run_data = {
    "timestamp": current_time,
    "bias_adjustments": bias_adjustments_arr.tolist() if hasattr(bias_adjustments_arr, 'tolist') else bias_adjustments_arr,
    "stdev_adjustments": stdev_adjustments_arr.tolist() if hasattr(stdev_adjustments_arr, 'tolist') else stdev_adjustments_arr,
    "true_theta": THETA_TRUE.flatten().tolist() if hasattr(THETA_TRUE, 'tolist') else THETA_TRUE.flatten(),
    "target_ess": target_ess,
    "num_mcmc_samples": num_mcmc_samples,
    "num_particles": NUM_PARTICLES,
    "add_noise_stdev": STD_DEV,
    "wasserstein_distance": wasser_dist,
    "posterior_mean_distance": post_mean_difference.flatten().tolist(),
    "Extra details": 'Priors: [uniform(0.001, 2), uniform(0, 4)]\n Used exp_3d HF Model'
}

save_run_hyperparameters(
    log_filename,
    run_label,
    **current_run_data
)

print(f"Run label: {run_label}")