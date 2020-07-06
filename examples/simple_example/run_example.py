import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import time

from copy import copy
from scipy.optimize import minimize
from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_translator import VectorMCMCTranslator
from smcpy import SMCSampler

from model import Model


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

    # run and time smcpy
    num_particles = 1000
    num_steps = 500
    num_mcmc_samples = 2
    ess_threshold = 0.75
    phi_sequence = np.linspace(0, 1, num_steps)
    mcmc_kernel = VectorMCMCTranslator(vector_mcmc, param_order=('a', 'b'))
    smc = SMCSampler(mcmc_kernel)

    time0 = time.time()
    step_list = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                           ess_threshold)
    time1 = time.time()

    mean = step_list[-1].compute_mean()
    evidence = smc.estimate_marginal_log_likelihood(step_list, phi_sequence)
    print('smc mean = {}'.format(mean))
    print('smc marginal log like = {}'.format(evidence))
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
                                   num_samples, cov, adapt_interval,
                                   adapt_delay=burnin)
    time1 = time.time()

    mean = chain[:, :, burnin:].mean(axis=2).mean(axis=0)
    plot_mcmc_chain(chain, ['a', 'b'], burnin=burnin)
    stacked_chain = np.hstack(chain[:, :, burnin:]).T
    log_likes = vector_mcmc.evaluate_log_likelihood(stacked_chain)
    mll = 1 / (np.sum(1 / np.exp(log_likes)) / len(log_likes))
    print('mcmc mean = a: {}, b: {}'.format(mean[0], mean[1]))
    print('mcmc marginal log like = {}'.format(np.log(mll)))
    print('total mcmc time = {}'.format(time1 - time0))

    # run and time smc w/ pymc3
    import pymc3 as pm
    model = pm.Model()
    with model:
        a = pm.Uniform('a', 0, 6)
        b = pm.Uniform('b', 0, 6)
        mu = a * x + b
        obs = pm.Normal('obs', mu=mu, sigma=std_dev, observed=noisy_data)

    time0 = time.time()
    with model:
        trace = pm.sample_smc(draws=num_particles, n_steps=num_mcmc_samples,
                              kernel='metropolis', parallel=False,
                              tune_steps=False, threshold=ess_threshold)
    time1 = time.time()

    mll = model.marginal_log_likelihood
    mean_a = np.mean(trace.get_values('a'))
    mean_b = np.mean(trace.get_values('b'))
    print('smc pymc3 num_steps = {}'.format(trace._report._n_steps))
    print('smc pymc3 mean = a: {}, b: {}'.format(mean_a, mean_b))
    print('smc pymc3 marginal log like = {}'.format(mll))
    print('total smc pymc3 time = {}'.format(time1 - time0))
