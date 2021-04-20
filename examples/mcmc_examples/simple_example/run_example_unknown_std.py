import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.utils.plotter import plot_mcmc_chain
from run_example_known_std import *


if __name__ == '__main__':

    np.random.seed(200)

    noisy_data = generate_data(eval_model, plot=False)

    num_samples = 10000
    burnin = 5000
    std_dev = None # estimate it; requires mod to init_inputs, cov, and priors
    init_inputs = np.array([[1, 1, 1], [2, 2, 2]])
    cov = np.eye(3)

    priors = [uniform(0., 6.), uniform(0., 6.), uniform(0, 10)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)

    chain = vector_mcmc.metropolis(init_inputs, num_samples, cov,
                                   adapt_interval=200, adapt_delay=100,
                                   progress_bar=True)

    plot_mcmc_chain(chain, param_labels=['a', 'b', 'std'], burnin=burnin,
                    include_kde=True)
