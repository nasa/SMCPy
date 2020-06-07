import numpy as np
import pymc3 as pm
import time

from copy import copy
from scipy.optimize import minimize

from smcpy.mcmc.pymc3_step_methods import SMCMetropolis
from smcpy.mcmc.pymc3_translator import PyMC3Translator
from smcpy.smc.initializer import Initializer
from smcpy.smc.mutator import Mutator

from model import Model

def perform_smc_sampling(num_particles, num_steps, num_mcmc_samples,
                         phi_sequence, mcmc_kernel):

    initializer = Initializer(mcmc_kernel, phi_sequence[1])
    mutator = Mutator(mcmc_kernel)

    smc_step = initializer.initialize_particles_from_prior(num_particles)
    step_list = [smc_step.copy()]
    for phi in phi_sequence[2:]:
        smc_step.update_weights(phi)
        smc_step.resample_if_needed(ess_threshold=num_particles * 0.75)
        smc_step = mutator.mutate(smc_step, num_mcmc_samples, phi)
        step_list.append(smc_step.copy())

    print('smc mean = {}'.format(smc_step.get_mean()))


if __name__ == '__main__':

    np.random.seed(100)

    # instance model / set up ground truth / add noise
    x = np.arange(100)
    eval_model = lambda a, b: a * x + b
    std_dev = 0.1
    y_true = eval_model(a=2, b=3.5)
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)

    # setup pymc3 model
    pymc3_model = pm.Model()
    with pymc3_model:
        a = pm.Uniform('a', 0., 5., transform=None)
        b = pm.Uniform('b', 0., 5., transform=None)
        #std_dev = pm.Uniform('std_dev', 0, 5, transform=None)
        mu = eval_model(a, b)
        obs = pm.Normal('obs', mu=mu, sigma=std_dev, observed=noisy_data)


    # set smc params
    num_particles = 200
    num_steps = 10
    num_mcmc_samples = 1
    phi_sequence = np.linspace(0, 1, num_steps)
    orig_mcmc_kernel = PyMC3Translator(pymc3_model, SMCMetropolis)

    ## run and time vanilla mcmc
    #time0 = time.time()
    #with pymc3_model:
    #    num_samples = num_particles * num_steps * num_mcmc_samples
    #    trace = pm.sample(num_samples, chains=1, cores=1, progressbar=False)
    #    a_mean = np.mean(trace.get_values('a'))
    #    b_mean = np.mean(trace.get_values('b'))
    #time1 = time.time()
    #print('mcmc mean = {}'.format({'a': a_mean, 'b': b_mean}))
    #print('total mcmc time = {}'.format(time1 - time0))

    # run and time mcmc with custom smc step method
    time0 = time.time()
    num_samples = num_particles * num_steps * num_mcmc_samples
    init_params = {'a': 1, 'b': 2}
    mcmc_kernel = copy(orig_mcmc_kernel)
    mcmc_kernel.sample(num_samples, init_params, cov=1, phi=1)
    a_mean = np.mean(mcmc_kernel._last_trace.get_values('a'))
    b_mean = np.mean(mcmc_kernel._last_trace.get_values('b'))
    time1 = time.time()
    print('mcmc smcstep mean = {}'.format({'a': a_mean, 'b': b_mean}))
    print('total mcmc smcstep time = {}'.format(time1 - time0))

    # run and time smc
    time0 = time.time()
    perform_smc_sampling(num_particles, num_steps, num_mcmc_samples,
                         phi_sequence, copy(orig_mcmc_kernel))
    time1 = time.time()
    print('total smc time = {}'.format(time1 - time0))

    ## profile smc
    #import cProfile, pstats, io
    #pr = cProfile.Profile()
    #pr.enable()
    #perform_smc_sampling(num_particles, num_steps, num_mcmc_samples,
    #                     phi_sequence, pymc3_model)
    #pr.disable()
    #s = io.StringIO()
    #ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    #ps.print_stats()
    #print(s.getvalue())
