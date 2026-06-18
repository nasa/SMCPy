import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json

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

##########################################################################################
##########################################################################################
##########################################################################################
'''
Run details
'''
# Data generation details
STD_DEV = 0.5
theta_0 = 1/20
theta_1 = 1
THETA_TRUE = np.array([[theta_0, theta_1]])
NUM_PARTICLES = 10_000
np.random.seed(42)
noisy_data = generate_noisy_data(THETA_TRUE, STD_DEV)


run_label = 'figs_ill_posed/test2'
bias_adjustments_arr = np.array([0.0005,0.005])
stdev_adjustments_arr = np.array([1,1])
target_ess = 0.9
num_mcmc_samples = 5
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

# # Setup low-fidelity case
priors = [uniform(0.001, 2), uniform(0, 4)]
vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)

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
np.save('pseudo_lofi_posterior.npy', lofi_particles)

# read in pseudo lofi posterior particles
lofi_particles = np.load('pseudo_lofi_posterior.npy')

perturbed_particles = perturb_lofi_posterior(lofi_particles, bias_adjustments_arr, stdev_adjustments_arr)

# Setup low-fidelity posterior as proposal for high-fidelity
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
    "num_particles": NUM_PARTICLES
}

# 2. Load existing data if the file already exists
if os.path.exists(log_filename):
    with open(log_filename, "r") as log_file:
        try:
            all_runs = json.load(log_file)
        except json.JSONDecodeError:
            all_runs = {} # If the file is empty or corrupted, start fresh
else:
    all_runs = {}

# 3. Add or Overwrite the data for this specific run_label
# (If run_label already exists in the dictionary, this overwrites it. If not, it adds it.)
all_runs[run_label] = current_run_data

# 4. Save the updated dictionary back to the JSON file
with open(log_filename, "w") as log_file:
    # indent=4 makes the JSON file nicely formatted and easy for humans to read!
    json.dump(all_runs, log_file, indent=4)

print(f"Run details successfully logged to {log_filename}")