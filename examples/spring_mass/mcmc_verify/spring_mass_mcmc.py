import numpy as np
from smcpy.smc.smc_sampler import SMCSampler
from mcmcpy.mcmc.mcmc_sampler import MCMCSampler
import sys
sys.path.append('../')
from spring_mass_models import SpringMassModel

# Initialize model
state0 = [0., 0.]                        #initial conditions
measure_t_grid = np.arange(0., 5., 0.2)  #time 
model = SpringMassModel(state0, measure_t_grid)

# Load data
noise_stddev = 0.94
displacement_data = np.genfromtxt('../noisy_data.txt')

# Define prior distributions
param_priors = {'K': ['Uniform', 0.0, 20.0],
                'g': ['Uniform', 0.0, 20.0]}

# SMC sampling
num_samples = 100000
num_samples_burnin = 50000
mcmc = MCMCSampler(displacement_data, model, param_priors)
mcmc.generate_pymc_model(std_dev0=noise_stddev, fix_var=True)
mcmc.sample(num_samples, num_samples_burnin)
mcmc.plot_pairwise(keys=['K', 'g'])
mcmc.plot_pdf(keys=['K', 'g'])
