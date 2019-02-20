import pytest
from smcpy.hdf5.hdf5_storage import HDF5Storage
import h5py
import numpy as np
import os
from smcpy.particles.smc_step import SMCStep
from smcpy.particles.particle import Particle


@pytest.fixture
def particle():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return particle


@pytest.fixture
def filled_step(particle):
    step_tester = SMCStep()
    step_tester.fill_step(5 * [particle])
    return step_tester


@pytest.fixture
def step_list(filled_step):
    return 3 * [filled_step]


@pytest.fixture
def h5file():
    return HDF5Storage('temp.hdf5', 'w')


def test_write_particle(h5file, particle):
    h5file.write_particle(particle, step_index=1, particle_index=1)
    os.remove('temp.hdf5')


def test_write_step(h5file, filled_step):
    h5file.write_step(filled_step, 1)
    f = h5py.File('temp.hdf5', 'r')
    f.get('steps/step_001/particle_0/weight')
    os.remove('temp.hdf5')


def test_write_step_list(h5file, step_list):
    h5file.write_step_list(step_list)
    f = h5py.File('temp.hdf5', 'r')
    f.get('steps/step_001/particle_0/weight')
    os.remove('temp.hdf5')


def test_read_particle(h5file, step_list):
    h5file.write_step_list(step_list)
    particle = h5file.read_particle(step_index=1, particle_index=1)
    assert particle.weight == 0.2
    os.remove('temp.hdf5')
