from smcpy.smc.smc_sampler import SMCSampler
from spring_mass_model import SpringMassModel
import numpy as np
import pytest
import h5py
import os
import warnings

@pytest.fixture
def model():
    state0 = [0., 0.]  # initial conditions
    measure_t_grid = np.arange(0., 5., 0.2)  # time
    model = SpringMassModel(state0, measure_t_grid)
    return model

@pytest.fixture
def sampler(model):
    displacement_data = np.genfromtxt('noisy_data.txt')
    param_priors = {'K': ['Uniform', 0.0, 10.0],
                    'g': ['Uniform', 0.0, 10.0]}
    sampler = SMCSampler(displacement_data, model, param_priors)
    return sampler

def test_autosaver(sampler):
    num_particles = 100
    num_time_steps = 10
    num_mcmc_steps = 1
    noise_stddev = 0.5
    step_list = sampler.sample(num_particles, num_time_steps, num_mcmc_steps,
                           noise_stddev, ess_threshold=num_particles * 0.5,
                           autosave_file='autosaver.hdf5')
    with h5py.File('autosaver.hdf5', 'r') as hdf:
        base_items = list(hdf.items())
        print("Items in base directory", base_items)
        group1 = hdf.get("steps")
        group1_items = list(group1.items())
        print group1_items
        assert len(group1_items) == num_time_steps

def test_load_step_list(sampler):
    num_time_steps = 10
    step_list = sampler.load_step_list('autosaver.hdf5')
    assert len(step_list) == num_time_steps

def test_restart_sampling(sampler):
    num_particles = 100
    num_time_steps = 10
    restart_time_step = 7
    num_mcmc_steps = 1
    noise_stddev = 0.5
    step_list = sampler.sample(num_particles, num_time_steps, num_mcmc_steps,
                               noise_stddev, restart_time_step=restart_time_step,
                               hdf5_to_load = 'autosaver.hdf5',
                               autosave_file = 'restart.hdf5')
    with h5py.File('restart.hdf5', 'r') as hdf:
        base_items = list(hdf.items())
        print("Items in base directory", base_items)
        group1 = hdf.get("steps")
        group1_items = list(group1.items())
        print group1_items
        assert len(group1_items) == num_time_steps
    os.remove('autosaver.hdf5')
    os.remove('restart.hdf5')
        
