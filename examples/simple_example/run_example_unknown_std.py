import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import uniform

from smcpy import SMCSampler
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy.utils.plotter import plot_pairwise

from run_example_known_std import *


if __name__ == '__main__':

    np.random.seed(200)

    noisy_data = generate_data(eval_model, plot=False)

    std_dev = None # estimate it; requires prior for std_dev and param_order

    priors = [uniform(0., 6.), uniform(0., 6.), uniform(0, 10)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=('a', 'b', 'std'))

    smc = SMCSampler(mcmc_kernel)
    phi_sequence = np.linspace(0, 1, 20)
    step_list, mll_list = smc.sample(num_particles=500, num_mcmc_samples=5,
                                     phi_sequence=phi_sequence,
                                     ess_threshold=1.0, progress_bar=True)

    print('marginal log likelihood = {}'.format(mll_list[-1]))
    print('parameter means = {}'.format(step_list[-1].compute_mean()))

    plot_pairwise(step_list[-1].params, step_list[-1].weights,
                  ['a', 'b', 'std'])
