import pytest
from smcpy.particles.particle_chain import ParticleChain


@pytest.fixture
def chain_tester():
    return ParticleChain()
