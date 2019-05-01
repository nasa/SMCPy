import h5py
import os


def test_autosaver(sampler):
    num_particles = 5
    num_time_steps = 3
    num_mcmc_steps = 1
    noise_stddev = 0.5
    sampler.sample(num_particles, num_time_steps, num_mcmc_steps,
                   noise_stddev, ess_threshold=num_particles * 0.5,
                   autosave_file='autosaver.hdf5')
    with h5py.File('autosaver.hdf5', 'r') as hdf:
        group1 = hdf.get("steps")
        group1_items = list(group1.items())
        assert len(group1_items) == num_time_steps


def test_load_step_list(sampler):
    num_time_steps = 3
    step_list = sampler.load_step_list('autosaver.hdf5')
    assert len(step_list) == num_time_steps


def test_trim_step_list(sampler):
    restart_time_step = 3
    step_list = sampler.load_step_list('autosaver.hdf5')
    trimmed_list = sampler.trim_step_list(step_list, restart_time_step)
    assert len(trimmed_list) == 3


def test_restart_sampling(sampler):
    num_particles = 5
    num_time_steps = 3
    restart_time_step = 2
    num_mcmc_steps = 1
    noise_stddev = 0.5
    sampler.sample(num_particles, num_time_steps, num_mcmc_steps,
                   noise_stddev, restart_time_step=restart_time_step,
                   hdf5_to_load='autosaver.hdf5',
                   autosave_file='restart.hdf5')
    with h5py.File('restart.hdf5', 'r') as hdf:
        group1 = hdf.get("steps")
        group1_items = list(group1.items())
        assert len(group1_items) == num_time_steps
    os.remove('autosaver.hdf5')
    os.remove('restart.hdf5')
