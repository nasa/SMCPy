import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import time

from copy import copy
from scipy.optimize import minimize
from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_translator import VectorMCMCTranslator
from smcpy.smc.initializer import Initializer
from smcpy.smc.mutator import Mutator
from smcpy.smc.updater import Updater

from model import Model


def perform_smc_sampling(num_particles, num_steps, num_mcmc_samples,
                         phi_sequence, mcmc_kernel):

    initializer = Initializer(mcmc_kernel, phi_sequence[1])
    mutator = Mutator(mcmc_kernel)
    updater = Updater(ess_threshold=0.75)

    particles = initializer.init_particles_from_prior(num_particles)
    step_list = [particles]
    for i, phi in enumerate(phi_sequence[2:]):
        particles = updater.update(particles, phi - phi_sequence[i + 1])
        particles = mutator.mutate(particles, phi, num_mcmc_samples)
        step_list.append(particles)

    print(particles.compute_covariance())
    print('smc mean = {}'.format(particles.compute_mean()))
    import pdb; pdb.set_trace()
    return step_list


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), '-k')
    ax.plot(x.flatten(), noisy_data.flatten(), 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plot_mcmc_chain(chain, param_names, burnin=0):
    fig, ax = plt.subplots(len(param_names))
    chain = chain[:, :, burnin:]
    for i, name in enumerate(param_names):
        for parallel_chain in chain:
            ax[i].plot(parallel_chain[i], '-')
        ax[i].set_ylabel(name)
    ax[-1].set_xlabel('sample #')
    plt.show()


if __name__ == '__main__':

    np.random.seed(100)

    # instance model / set up ground truth / add noise
    x = np.arange(100)

    def eval_model(theta):
        a = theta[:, 0, None]
        b = theta[:, 1, None]
        return a * x + b

    std_dev = 2
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    plot_noisy_data(x, y_true, noisy_data)

    # setup vector mcmc model
    priors = [uniform(0., 6.), uniform(0., 6.)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)

    # set smc params
    num_particles = 1000
    num_steps = 25
    num_mcmc_samples = 5
    phi_sequence = np.linspace(0, 1, num_steps)
    #phi_sequence = (np.exp(5 * phi_sequence/num_steps) - 1) / (np.exp(5) - 1)

    # set up mcmc kernel translator
    mcmc_kernel = VectorMCMCTranslator(vector_mcmc, param_order=('a', 'b'))

    # run and time smc
    time0 = time.time()
    step_list = perform_smc_sampling(num_particles, num_steps, num_mcmc_samples,
                                     phi_sequence, mcmc_kernel)
    time1 = time.time()
    print('total smc time = {}'.format(time1 - time0))

    # run and time mcmc
    num_parallel_chains = 4
    num_samples = int(num_particles * num_steps * num_mcmc_samples /
                      num_parallel_chains)
    burnin = int(num_samples / 3)
    cov = np.array([[1, 0], [0, 1]])
    adapt_interval = 100

    prior_samples = [prior.rvs(num_parallel_chains) for prior in priors]
    time0 = time.time()
    chain = vector_mcmc.metropolis(np.array(prior_samples).T,
                                   num_samples, cov, adapt_interval)
    time1 = time.time()
    mean = chain[:, :, burnin:].mean(axis=2).mean(axis=0)
    plot_mcmc_chain(chain, ['a', 'b'], burnin=burnin)
    print('mcmc mean = a: {}, b: {}'.format(mean[0], mean[1]))
    print('total mcmc time = {}'.format(time1 - time0))
