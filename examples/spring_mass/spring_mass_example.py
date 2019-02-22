import numpy as np
from spring_mass_model import SpringMassModel
from smcpy.smc.smc_sampler import SMCSampler

# Initialize model
state0 = [0., 0.]  # initial conditions
measure_t_grid = np.arange(0., 5., 0.2)  # time
model = SpringMassModel(state0, measure_t_grid)

# Load data
noise_stddev = 0.5
displacement_data = np.genfromtxt('noisy_data.txt')

# Define prior distributions
param_priors = {'K': ['Uniform', 0.0, 10.0],
                'g': ['Uniform', 0.0, 10.0]}

# SMC sampling
num_particles = 500
num_time_steps = 20
num_mcmc_steps = 1
smc = SMCSampler(displacement_data, model, param_priors)
pchain, step_list = smc.sample(num_particles, num_time_steps, num_mcmc_steps,
                               noise_stddev, ess_threshold=num_particles * 0.5)

# Calculate means
means = pchain.get_mean()
print '\n'
for key, value in means.iteritems():
    print '%s mean = %s' % (key, value)

# Plot
if smc._rank == 0:
    pchain.plot_pairwise_weights(save=True, show=False)
