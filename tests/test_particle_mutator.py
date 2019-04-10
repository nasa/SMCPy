import numpy as np
import pytest
from smcpy.smc.particle_mutator import ParticleMutator
from smcpy.mcmc.mcmc_sampler import MCMCSampler
from smcpy.utils.single_rank_comm import SingleRankComm
from smcpy.particles.particle import Particle
from smcpy.smc.smc_step import SMCStep


class DummyModel():
    '''Dummy test model'''

    def __init__(self):
        pass

    def evaluate(*args, **kwargs):
        return np.array([0., 0., 0.])


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return 5 * [particle]


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.set_particles(particle_list)
    return step_tester


@pytest.fixture
def mcmc_obj():
    model = DummyModel()
    data = model.evaluate({'a': 0., 'b': 0.})
    param_priors = {'a': ['Uniform', 0., 1.], 'b': ['Uniform', 0., 1.]}
    return MCMCSampler(data, model, param_priors, storage_backend='ram')


@pytest.fixture
def part_mutator(filled_step, mcmc_obj):
    return ParticleMutator(filled_step, mcmc_obj, num_mcmc_steps=10,
                           mpi_comm=SingleRankComm())
