import pytest
import numpy as np
from smcpy.particles.smc_step import SMCStep
from smcpy.particles.particle import Particle


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return 5 * [particle]


@pytest.fixture
def filled_step(particle_list):
    step_tester = SMCStep().fill_step(particle_list)
    return step_tester


@pytest.fixture
def step_list(filled_step):
    return 3 * [filled_step]
