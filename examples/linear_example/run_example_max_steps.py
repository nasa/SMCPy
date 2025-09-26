import numpy as np
import time

from scipy.stats import uniform

from smcpy import MaxStepSampler, VectorMCMCKernel, VectorMCMC


TRUE_PARAMS = np.array([[2, 3.5]])
TRUE_STD = 2


def eval_model(theta):
    time.sleep(0.1)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(eval_model):
    y_true = eval_model(TRUE_PARAMS)
    return y_true + np.random.normal(0, TRUE_STD, y_true.shape)


if __name__ == "__main__":
    np.random.seed(200)

    max_smc_steps = 10
    norm_step_threshold = 0.5
    std_dev = 2
    noisy_data = generate_data(eval_model)

    priors = [uniform(0.0, 6.0), uniform(0.0, 6.0)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=("a", "b"))

    smc = MaxStepSampler(
        mcmc_kernel, max_steps=max_smc_steps, norm_step_threshold=norm_step_threshold
    )
    step_list, mll_list = smc.sample(
        num_particles=500,
        num_mcmc_samples=5,
        target_ess=0.8,  # Using adaptive sampling parameters
    )
    print("marginal log likelihood = {}".format(mll_list[-1]))
    print("parameter means = {}".format(step_list[-1].compute_mean()))

    print(" Step | Phi Value")
    print("------|----------")
    for i, value in enumerate(smc.phi_sequence):
        print(f"{i:5d} | {value:5.3e}")
