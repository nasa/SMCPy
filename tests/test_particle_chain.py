import pytest
from smcpy.particles.particle_chain import ParticleChain
from smcpy.particles.particle import Particle


@pytest.fixture
def particle_chain():
    return ParticleChain()


@pytest.fixture
def zero_weight_particle():
    weight = 0
    params = {'a': 1.}
    log_like = -1e-2
    return Particle(params, weight, log_like)
    

def test_total_weight_is_zero_value_error(particle_chain, zero_weight_particle):
    particle_chain.add_step([zero_weight_particle, zero_weight_particle])
    with pytest.raises(ValueError):
        particle_chain.normalize_step_weights()
