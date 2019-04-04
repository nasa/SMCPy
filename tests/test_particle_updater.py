import numpy as np
import pytest
from smcpy.smc.particle_initializer import ParticleUpdater
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle


class DummyComm():
    '''Dummy mpi communicator'''

    def __init__(self):
        pass

    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0


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
def part_updater(filled_step):
    return ParticleUpdater(filled_step, mpi_comm=DummyComm())


def test_update_particles():
    assert True
