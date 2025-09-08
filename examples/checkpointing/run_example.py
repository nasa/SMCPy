import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import time

from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import AdaptiveSampler
from smcpy.utils.storage import HDF5Storage as Storage


def eval_model(theta):
    time.sleep(0.1)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.default_rng(34).normal(0, std_dev, y_true.shape)
    return noisy_data


def kill_after_timeout(run_smc, timeout_seconds):
    process = multiprocessing.Process(target=run_smc)
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        print(f"\n>> Killing run_pysips after {timeout_seconds} seconds.")
        process.terminate()
        process.join()
    else:
        print("Function completed naturally.")


if __name__ == "__main__":

    checkpoint_file = "example.h5"

    num_smc_steps = 5
    std_dev = 2
    noisy_data = generate_data(eval_model, std_dev, plot=False)

    priors = [uniform(0.0, 6.0), uniform(0.0, 6.0)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(
        vector_mcmc, param_order=("a", "b"), rng=np.random.default_rng(34)
    )

    # Setup function for running SMC w/ storage context
    def run_smc(mode="w"):  # mode = 'w' will create a new checkpoint file
        with Storage(checkpoint_file, mode=mode):  # can use any available backend
            smc = AdaptiveSampler(mcmc_kernel)
            results, mll = smc.sample(
                num_particles=500, num_mcmc_samples=5, target_ess=0.8
            )
        return results, mll

    # Start run but kill after 7 seconds (incomplete run)
    kill_after_timeout(run_smc, timeout_seconds=7)

    # Restart SMC run and allow run to complete
    print(">> Restarting SMC run...")
    results, mll = run_smc(mode="a")

    # Print results to console
    print("marginal log likelihood = {}".format(mll[-1]))
    print("parameter means = {}".format(results[-1].compute_mean()))
    print("phi sequence = {}".format(np.array(results.phi_sequence)))

    # Show that SMC will not run again once complete (if mode != 'w')
    print(">> Try restarting again...")
    results, mll = run_smc(mode="a")

    # Print results to console
    print("marginal log likelihood = {} (unchanged)".format(mll[-1]))
