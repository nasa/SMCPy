from model import model
import numpy as np
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
    y_noisy = np.genfromtxt('noisy_data.txt')
    
    param_priors = {'a': ['Uniform', -5.0, 5.0],
                     'b': ['Uniform', -5.0, 5.0]}
    
    # run smc
    num_particles = 1000
    num_time_steps = 20
    num_mcmc_steps = 5
    smc = SMCSampler(y_noisy, m, param_priors)
    particle_chain = smc.sample(num_particles, num_time_steps, num_mcmc_steps,
                                std_dev, ESS_threshold=0.5*num_particles,
                                autosave_file='smc.h5')

    # try a restart
    restarted_chain = smc.sample(num_particles, num_time_steps, num_mcmc_steps,
                                 std_dev, ESS_threshold=0.5*num_particles,
                                 restart_time_step=10, hdf5_to_load='smc.h5',
                                 autosave_file='smc_restart.h5')

    # plot
    try:
        particle_chain.plot_pairwise_weights(save=True, show=False)
        restarted_chain.plot_pairwise_weights(save=True, show=False,
                                              prefix='pairwise_restart')
    except:
        pass
