from model import model
import os
import time
from mcmc.MCMC import MCMC
import numpy as np
import pickle
from smcpy.particles.particle import Particle
from smcpy.smc.smc_sampler import SMCSampler
from scipy.optimize import minimize


if __name__ == '__main__':
    # instance model / set up ground truth / add noise
    a = 2
    b = 3.5
    x = np.arange(50)
    m = model(x)
    std_dev = 0.6
    y_true = m.evaluate(a, b)
    y_noisy = y_true + np.random.normal(0, std_dev, y_true.shape)
    
    param_priors = {'a': ['Uniform', -5.0, 5.0],
                     'b': ['Uniform', -5.0, 5.0]}
    
    # get optimal params (LSR)
    initial_guess = {'a': 2.0, 'b': 3.5}
    mcmc = MCMC(data=y_noisy, model=m, params=param_priors)
    params_opt, ssq_opt = mcmc.fit(initial_guess, opt_method='nelder-mead')
    center = params_opt
    scales = {'a': 0.01, 'b': 0.01}
        
    # run smc
    num_particles = 1000
    num_time_steps = 10
    num_mcmc_steps = 2
    smc = SMCSampler(y_noisy, m, param_priors)
    particle_chain = smc.sample(num_particles, num_time_steps, num_mcmc_steps,
                                std_dev, ESS_threshold=0.5*num_particles,
                                proposal_center=center, proposal_scales=scales,
                                hdf5_file_path='test.hdf5')
    particle_chain.plot_marginal('a', step=0)
    particle_chain.plot_marginal('a', step=1)
