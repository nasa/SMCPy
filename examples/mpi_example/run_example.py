import matplotlib.pyplot as plt
import numpy as np
import pandas
import pymc3 as pm
import sys
import time

from copy import copy
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler

from model import Model


def gen_noisy_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), '-k')
    ax.plot(x.flatten(), noisy_data.flatten(), 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':

    np.random.seed(200)

    x = np.arange(100)
    def eval_model(theta):
        a = theta[:, 0, None]
        b = theta[:, 1, None]
        return a * x + b

    std_dev = 2
    noisy_data = gen_noisy_data(eval_model, std_dev, plot=False)

    # set analysis params
    num_repeats = 500
    n_processors = 2

    # set smc params
    num_particles = 500
    num_mcmc_samples = 10
    ess_threshold = 0.8
    priors = [uniform(0., 6.), uniform(0., 6.)]

    parallel_mcmc = ParallelMCMC(eval_model, noisy_data, priors, std_dev)

    phi_sequence = np.linspace(0, 1, num_smc_steps)

    mcmc_kernel = VectorMCMCKernel(parallel_mcmc, param_order=('a', 'b'))
    step_list, evidence = SMCSampler(mcmc_kernel)

    print(step_list[-1].get_mean())
