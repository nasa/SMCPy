'''
Contains all fixtures required in tests
'''
import pytest
import numpy as np
from particle_tester import ParticleTester
from smcpy.smc.particle_initializer import ParticleInitializer
from smcpy.smc.particle_mutator import ParticleMutator
from smcpy.smc.particle_updater import ParticleUpdater
from smcpy.smc.smc_sampler import SMCSampler
from smcpy.mcmc.mcmc_sampler import MCMCSampler
from smcpy.utils.single_rank_comm import SingleRankComm
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle
from smcpy.hdf5.hdf5_storage import HDF5Storage


class DummyModel():
    '''Dummy test model'''

    def __init__(self):
        pass

    def evaluate(*args, **kwargs):
        return np.array([0., 0., 0.])


@pytest.fixture
def particle_tester():
    return ParticleTester()


@pytest.fixture
def particle():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return particle


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return [particle.copy() for i in range(5)]


@pytest.fixture
def mixed_particle_list():
    particle_1 = Particle({'a': 1, 'b': 2}, -0.2, -0.2)
    particle_2 = Particle({'a': 2, 'b': 4}, -0.2, -0.2)
    list_1 = [particle_1.copy() for i in range(3)]
    list_2 = [particle_2.copy() for i in range(2)]
    return list_1 + list_2


@pytest.fixture
def linear_particle_list():
    a = np.arange(0, 20)
    b = [2 * val + np.random.normal(0, 1) for val in a]
    w = 0.1
    p_list = [Particle({'a': a[i], 'b': b[i]}, w, -.2) for i in a]
    return p_list


@pytest.fixture
def mixed_weight_particle_list():
    p_list = [Particle({'a': 1, 'b': 2}, i, -.2) for i in np.arange(.1, .5, .1)]
    return p_list


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.set_particles(particle_list)
    return step_tester


@pytest.fixture
def step_list(filled_step):
    return 3 * [filled_step]


@pytest.fixture
def mixed_step(step_tester, mixed_particle_list):
    step_tester.set_particles(mixed_particle_list)
    return step_tester


@pytest.fixture
def linear_step(step_tester, linear_particle_list):
    step_tester.set_particles(linear_particle_list)
    return step_tester


@pytest.fixture
def mixed_weight_step(step_tester, mixed_weight_particle_list):
    step_tester.set_particles(mixed_weight_particle_list)
    return step_tester


@pytest.fixture
def mcmc_obj():
    model = DummyModel()
    data = model.evaluate({'a': 0., 'b': 0.})
    param_priors = {'a': ['Uniform', 0., 1.], 'b': ['Uniform', 0., 1.]}
    return MCMCSampler(data, model, param_priors, storage_backend='ram')


@pytest.fixture
def part_initer(mcmc_obj):
    return ParticleInitializer(mcmc_obj, temp_schedule=[0., 1.],
                               mpi_comm=SingleRankComm())


@pytest.fixture
def part_mutator(filled_step, mcmc_obj):
    return ParticleMutator(filled_step, mcmc_obj, num_mcmc_steps=10,
                           mpi_comm=SingleRankComm())


@pytest.fixture
def part_updater(filled_step):
    return ParticleUpdater(filled_step, 2.5, mpi_comm=SingleRankComm())


@pytest.fixture
def part_updater_high_ess_threshold(filled_step):
    return ParticleUpdater(filled_step, 10, mpi_comm=SingleRankComm())


@pytest.fixture
def sampler():
    model = DummyModel()
    param_priors = {'a': ['Uniform', 0., 1.], 'b': ['Uniform', 0., 1.]}
    data = model.evaluate({'a': 0., 'b': 0.})
    sampler = SMCSampler(data, model, param_priors)
    return sampler


@pytest.fixture
def h5file():
    return HDF5Storage('temp.hdf5', 'w')


@pytest.fixture
def filled_h5file(h5file, step_list):
    h5file.write_step_list(step_list)
    return h5file
