import os
import sys
import argparse
import datetime
import tomllib  # Use 'import tomli as tomllib' if on Python 3.10 or older
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, wasserstein_distance

# Set up paths
sys.path.append(os.path.abspath(os.path.join("/u/vasanche/SMCPy")))
sys.path.append(os.path.abspath(os.path.join("/u/vasanche/SMCPy/smcpy")))

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler as Sampler
from smcpy.paths import GeometricPath
from smcpy.mfmc_proposal import MultiFidelityProposal

from examples.mf_smc.exp_3d import M_HF
from examples.mf_smc.plotting_helpers import (
    plot_ill_posed_res, 
    save_run_hyperparameters, 
    plot_param_hist_progression, 
    plot_target_boxplots
)

def perturb_lofi_posterior(particles, bias_adjust, stdev_adjust):
    """Adjusts the variance and bias of the particles."""
    mean_particles = np.mean(particles, axis=0)
    perturbed_particles = (particles - mean_particles) * stdev_adjust + mean_particles  
    perturbed_particles = perturbed_particles + bias_adjust
    return perturbed_particles

def main(config_file):
    # Load configuration from TOML
    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    # Extract parameters
    paths = config["paths"]
    model_params = config["model_parameters"]
    smc_params = config["smc_parameters"]
    adjustments = config["adjustments"]

    STD_DEV = model_params["std_dev"]
    THETA_TRUE = np.array([[model_params["theta_0"], model_params["theta_1"]]])
    NUM_PARTICLES = smc_params["num_particles"]
    
    bias_adjustments_arr = np.array(adjustments["bias"])
    stdev_adjustments_arr = np.array(adjustments["stdev"])

    # Load data
    noisy_data = np.load(paths["noisy_data_file"])
    lofi_particles = np.load(paths["lofi_particles_file"])

    # Define pseudo Low-Fidelity model
    def pseudo_M_LF(THETA, reference_mean):
        """
        Evaluates the Low-Fidelity likelihood by fully mapping the particles 
        back to the High-Fidelity space (reversing both bias and variance).
        """
        # 1. Reverse the bias (Step 4 reversed)
        unbiased_THETA = THETA - bias_adjustments_arr
        
        # 2. Reverse the standard deviation stretch/shrink (Steps 1-3 reversed)
        # Subtract the mean, divide by the scale factor, then add the mean back
        mapped_THETA = ((unbiased_THETA - reference_mean) / stdev_adjustments_arr) + reference_mean
        
        # 3. Evaluate using the true High-Fidelity model
        return M_HF(mapped_THETA)

    # Apply perturbations
    perturbed_particles = perturb_lofi_posterior(
        lofi_particles, bias_adjustments_arr, stdev_adjustments_arr
    )
    mean_particles = np.mean(lofi_particles, axis=0)
    my_LF_model_for_SMC = lambda THETA: pseudo_M_LF(THETA, reference_mean=mean_particles)

    # Setup priors and proposal
    priors = [uniform(0.001, 2), uniform(0, 4)]
    lofi_proposal_dist = MultiFidelityProposal(
        perturbed_particles, 
        my_LF_model_for_SMC, 
        noisy_data,
        priors,
        STD_DEV
    )

    # Setup high-fidelity MCMC definition
    hifi_vector_mcmc = VectorMCMC(M_HF, noisy_data, priors, STD_DEV)
    mcmc_kernel = VectorMCMCKernel(
        hifi_vector_mcmc, 
        param_order=("theta_0", "theta_1"), 
        path=GeometricPath(proposal=lofi_proposal_dist)
    )

    # Run High-Fidelity SMC
    hifi_smc = Sampler(mcmc_kernel=mcmc_kernel, show_progress_bar=True)
    hifi_step_list, _ = hifi_smc.sample(
        num_particles=NUM_PARTICLES,
        num_mcmc_samples=smc_params["num_mcmc_samples"],
        target_ess=smc_params["target_ess"],
    )
    
    hifi_phi_list = hifi_smc.phi_sequence
    hifi_particles = hifi_step_list[-1].params

    # Plot results
    run_label = paths["run_label"]
    
    plot_ill_posed_res(
        THETA_TRUE.flatten(), run_label, perturbed_particles,
        bias_adjustments_arr, stdev_adjustments_arr,
        High_Fidelity=(hifi_step_list, hifi_phi_list)
    )

    plot_param_hist_progression(
        run_label, lofi_particles, perturbed_particles, hifi_particles,
        bias_adjustments_arr, stdev_adjustments_arr
    )

    plot_target_boxplots(
        THETA_TRUE.flatten(), run_label, 
        High_Fidelity=(hifi_step_list, hifi_phi_list)
    )

    # Calculate Wasserstein distances and mean differences
    wasser_dist = [
        wasserstein_distance(lofi_particles[:, i], hifi_particles[:, i]) 
        for i in range(lofi_particles.shape[-1])
    ]
    
    post_mean_difference = np.mean(hifi_particles, axis=0) - np.mean(lofi_particles, axis=0)

    # Log Run Information to JSON
    current_run_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bias_adjustments": bias_adjustments_arr.tolist(),
        "stdev_adjustments": stdev_adjustments_arr.tolist(),
        "true_theta": THETA_TRUE.flatten().tolist(),
        "target_ess": smc_params["target_ess"],
        "num_mcmc_samples": smc_params["num_mcmc_samples"],
        "num_particles": NUM_PARTICLES,
        "add_noise_stdev": STD_DEV,
        "wasserstein_distance": wasser_dist,
        "posterior_mean_distance": post_mean_difference.tolist(),
        "num_smc_steps": len(hifi_phi_list),
        "Extra details": "Priors: [uniform(0.001, 2), uniform(0, 4)]\n Used exp_3d HF Model"
    }

    save_run_hyperparameters(paths["log_filename"], run_label, **current_run_data)
    print(f"Run label: {run_label}")

if __name__ == "__main__":
    # Setup Argument Parser to read the TOML file from the command line
    parser = argparse.ArgumentParser(description="Run Multi-Fidelity SMC with a TOML config.")
    parser.add_argument("config", help="Path to the TOML configuration file.")
    args = parser.parse_args()

    # Run the main function
    main(args.config)