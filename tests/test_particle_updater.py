import numpy as np
import pytest
from smcpy.smc.particle_updater import ParticleUpdater
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle
from smcpy.utils.single_rank_comm import SingleRankComm


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
    return ParticleUpdater(filled_step, None, mpi_comm=SingleRankComm())


def test_update_log_weights(part_updater):
    temperature_step = 0.1
    exp_w = np.exp(0.2 - 0.2 * temperature_step)
    part_updater.update_log_weights(temperature_step)
    assert all([exp_w == p.log_weight for p in part_updater.step.get_particles()])


def test_resample_if_needed(part_updater):
    print part_updater.step.compute_ess()
    assert False
#
