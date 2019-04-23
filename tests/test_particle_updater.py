import numpy as np
import pytest
from smcpy.smc.particle_updater import ParticleUpdater
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle
from smcpy.utils.single_rank_comm import SingleRankComm


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    particle_list = [particle.copy() for i in range(5)]
    return particle_list


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.set_particles(particle_list)
    return step_tester


@pytest.fixture
def part_updater(filled_step):
    return ParticleUpdater(filled_step, 2.5, mpi_comm=SingleRankComm())


@pytest.fixture
def part_updater_high_ess_threshold(filled_step):
    return ParticleUpdater(filled_step, 10, mpi_comm=SingleRankComm())


def test_update_log_weights(part_updater):
    temperature_step = 0.1
    exp_w = 0.2 - 0.2 * temperature_step
    part_updater.update_log_weights(temperature_step)
    assert all([exp_w == p.log_weight for p in
                part_updater.step.get_particles()])


def test_resample_if_needed_resample(part_updater):
    part_updater.resample_if_needed()
    assert part_updater._resample_status == "No resampling"
