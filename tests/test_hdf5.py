import pytest
from smcpy.hdf5.hdf5_storage import HDF5Storage
import h5py
import os
import numpy as np
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle


def test_write_particle(h5file, particle):
    h5file.write_particle(particle, step_index=1, particle_index=1)
    f = h5py.File('temp.hdf5', 'r')
    assert np.array(f.get('steps/step_001/particle_1/log_weight')) == 0.2
    os.remove('temp.hdf5')


def test_write_step(h5file, filled_step):
    h5file.write_step(filled_step, 1)
    f = h5py.File('temp.hdf5', 'r')
    assert np.array(f.get('steps/step_001/particle_1/log_weight')) == 0.2
    os.remove('temp.hdf5')


def test_write_step_list(h5file, step_list):
    h5file.write_step_list(step_list)
    f = h5py.File('temp.hdf5', 'r')
    assert np.array(f.get('steps/step_001/particle_0/log_weight')) == 0.2
    os.remove('temp.hdf5')


def test_read_particle(filled_h5file):
    particle = filled_h5file.read_particle(step_index=1, particle_index=0)
    assert particle.log_weight == 0.2
    os.remove('temp.hdf5')


def test_read_step(filled_h5file):
    step = filled_h5file.read_step(step_index=1)
    assert np.array_equal(step.get_log_weights(), [0.2] * 5)
    os.remove('temp.hdf5')


def test_read_step_list(filled_h5file):
    step_list = filled_h5file.read_step_list()
    assert np.array_equal(step_list[0].get_log_weights(), [0.2] * 5)
    os.remove('temp.hdf5')


def test_close(h5file, step_list):
    h5file.close()
    with pytest.raises(ValueError):
        h5file.write_step_list(step_list)
    os.remove('temp.hdf5')


def test_get_num_steps(filled_h5file):
    num_steps = filled_h5file.get_num_steps()
    assert num_steps == 3
    os.remove('temp.hdf5')


def test_get_num_particles_in_step(filled_h5file):
    num_particles = filled_h5file.get_num_particles_in_step(step_index=1)
    assert num_particles == 5
    os.remove('temp.hdf5')


def test_create_step_group(h5file):
    h5file._create_step_group(1)
    assert h5file._h5.keys() == ['steps']
    os.remove('temp.hdf5')
