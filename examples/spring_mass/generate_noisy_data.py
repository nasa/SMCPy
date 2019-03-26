import numpy as np
from spring_mass_models import SpringMassModel

# Initialize model
state0 = [0., 0.]  # initial conditions
measure_t_grid = np.arange(0., 5., 0.2)  # time
model = SpringMassModel(state0, measure_t_grid)

# Get ground truth
true_params = {'K': 1.67, 'g': 4.62}
true_t_grid = np.arange(0., 10., 0.02)  # time
true_model = SpringMassModel(state0, true_t_grid)
true = true_model.evaluate(true_params)

# Add noise
noise_std_dev = 0.5
noisy_data = model.generate_noisy_data_with_model(noise_std_dev, true_params)

# Save data
np.savetxt('noisy_data.txt', noisy_data)

# Plot
import matplotlib.pyplot as plt
plt.plot(measure_t_grid, noisy_data, 'o')
plt.plot(true_t_grid, true, '-')
plt.xlabel('time')
plt.ylabel('displacement')
plt.savefig('noisy_data.png')
