from model import Model
import numpy as np
from smcpy.particles.particle import Particle
from smcpy.smc.smc_sampler import SMCSampler
from scipy.optimize import minimize


if __name__ == '__main__':
    # instance model / set up ground truth / add noise
    a = 2
    b = 3.5
    x = np.arange(50)
    my_model = Model(x)
    std_dev = None # measurement noise std deviation will be sampled
    noisy_data = np.genfromtxt('noisy_data.txt')

    param_priors = {'a': ['Uniform', -5.0, 5.0],
                    'b': ['Uniform', -5.0, 5.0]}

    # run smc
    num_particles = 1000
    num_time_steps = 20
    num_mcmc_steps = 2
    smc = SMCSampler(noisy_data, my_model, param_priors)
    step_list = smc.sample(num_particles, num_time_steps, num_mcmc_steps,
                           std_dev, ess_threshold=0.5 * num_particles,
                           autosave_file='test.h5')

    # plot results of last step
    try:
        step_list[-1].plot_pairwise_weights(show=False, save=True)
    except:
        pass
