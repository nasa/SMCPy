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


def run_and_time_smcpy(vector_mcmc, num_particles, num_mcmc_samples,
                       num_smc_steps, ess_threshold):

    phi_sequence = np.linspace(0, 1, num_smc_steps)

    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=('a', 'b'))
    smc = SMCSampler(mcmc_kernel)

    time0 = time.time()
    # hacked sample
    step_list, evidence = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                           ess_threshold)
    time1 = time.time()

    mean = step_list[-1].compute_mean()
    #evidence = smc.estimate_marginal_log_likelihood(step_list, phi_sequence)

    return mean, evidence, (time1 - time0)


def run_and_time_mcmc(priors, num_particles, num_smc_steps, num_mcmc_samples,
                      init_cov, adapt_interval, burnin_ratio, plot=True):

    num_samples = int(num_particles * num_smc_steps * num_mcmc_samples)
    burnin = int(burnin_ratio * num_samples)

    prior_samples = [prior.rvs(num_parallel_chains) for prior in priors]
    time0 = time.time()
    chain = vector_mcmc.metropolis(np.array(prior_samples).T,
                                   num_samples, init_cov, adapt_interval,
                                   adapt_delay=burnin)
    time1 = time.time()

    mean = chain[:, :, burnin:].mean(axis=2).mean(axis=0)
    if plot:
        plot_mcmc_chain(chain, ['a', 'b'], burnin=burnin)
    stacked_chain = np.hstack(chain[:, :, burnin:]).T
    log_likes = vector_mcmc.evaluate_log_likelihood(stacked_chain)
    evidence = 1 / (np.sum(1 / np.exp(log_likes)) / len(log_likes))

    return mean, np.log(evidence), (time1 - time0)

def run_and_time_pymc3smc(model, num_particles, ess_threshold,
                          num_mcmc_samples):
    time0 = time.time()
    with model:
        trace = pm.sample_smc(draws=num_particles, n_steps=num_mcmc_samples,
                              kernel='metropolis', parallel=False,
                              tune_steps=False, threshold=ess_threshold)
    time1 = time.time()

    evidence = model.marginal_log_likelihood
    mean_a = np.mean(trace.get_values('a'))
    mean_b = np.mean(trace.get_values('b'))

    return {'a': mean_a, 'b': mean_b}, evidence, (time1 - time0)


def generate_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def generate_vector_mcmc(eval_model, noisy_data, priors, std_dev):
    return VectorMCMC(eval_model, noisy_data, priors, std_dev)


def generate_pymc3_model(priors, std_dev):
    model = pm.Model()
    with model:
        a = pm.Uniform('a', *priors[0].support())
        b = pm.Uniform('b', *priors[1].support())
        mu = a * x + b
        obs = pm.Normal('obs', mu=mu, sigma=std_dev, observed=noisy_data)
    return model


def run_one_repeat(vector_mcmc, num_particles, num_mcmc_samples,
                   ess_threshold, priors, init_cov, adapt_interval,
                   burnin_ratio, pymc3_model):

    row_dict = {'num_smc_steps': [], 'smcpy_mean_a': [], 'smcpy_mean_b': [],
                'smcpy_mll': [], 'mcmc_mean_a': [], 'mcmc_mean_b': [],
                'mcmc_mll': [], 'pymc3_mean_a': [], 'pymc3_mean_b': [],
                'pymc3_mll': []}

    for num_smc_steps in np.arange(10, 110, 10):

        row_dict['num_smc_steps'].append(num_smc_steps)

        outputs = run_and_time_smcpy(vector_mcmc, num_particles,
                                     num_mcmc_samples, num_smc_steps,
                                     ess_threshold)

        row_dict['smcpy_mean_a'].append(outputs[0]['a'])
        row_dict['smcpy_mean_b'].append(outputs[0]['b'])
        row_dict['smcpy_mll'].append(outputs[1])

        outputs = run_and_time_mcmc(priors, num_particles, num_smc_steps,
                                    num_mcmc_samples, init_cov,
                                    adapt_interval, burnin_ratio, plot=False)

        row_dict['mcmc_mean_a'].append(outputs[0][0])
        row_dict['mcmc_mean_b'].append(outputs[0][1])
        row_dict['mcmc_mll'].append(outputs[1])

        outputs = run_and_time_pymc3smc(pymc3_model, num_particles,
                                        ess_threshold, num_mcmc_samples)

        row_dict['pymc3_mean_a'].append(outputs[0]['a'])
        row_dict['pymc3_mean_b'].append(outputs[0]['b'])
        row_dict['pymc3_mll'].append(outputs[1])

    return pandas.DataFrame(row_dict)



if __name__ == '__main__':

    np.random.seed(200)

    # set model and tools
    x = np.arange(100)
    def eval_model(theta):
        a = theta[:, 0, None]
        b = theta[:, 1, None]
        return a * x + b

    std_dev = 2
    noisy_data = generate_data(eval_model, std_dev, plot=False)

    # set analysis params
    num_repeats = 500
    n_processors = 10

    # set smc params
    num_particles = 500
    num_mcmc_samples = 10
    ess_threshold = 1.
    priors = [uniform(0., 6.), uniform(0., 6.)]

    # set mcmc params
    num_parallel_chains = 1
    burnin_ratio = 1/3
    init_cov = np.array([[1, 0], [0, 1]])
    adapt_interval = 100

    # generate stoch models
    vector_mcmc = generate_vector_mcmc(eval_model, noisy_data, priors, std_dev)
    pymc3_model = generate_pymc3_model(priors, std_dev)

    # run samplers in parallel varying smc steps
    pool = Pool(n_processors)
    args = (vector_mcmc, num_particles, num_mcmc_samples, ess_threshold, priors,
            init_cov, adapt_interval, burnin_ratio, pymc3_model)
    res = [pool.apply_async(run_one_repeat, args) for i in range(num_repeats)]
    pool.close()
    pool.join()

    df = pandas.concat([r.get() for r in res]).reset_index(drop=True)
    df.to_hdf('sampling_data.h5', key='data')
