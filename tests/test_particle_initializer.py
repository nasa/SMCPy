import numpy as np
import pytest
from smcpy.smc.particle_initializer import ParticleInitializer
from smcpy.mcmc.mcmc_sampler import MCMCSampler


class DummyModel():
    '''Dummy test model'''

    def __init__(self):
        pass

    def evaluate(*args, **kwargs):
        return np.array([0., 0., 0.])


class DummyComm():
    '''Dummy mpi communicator'''

    def __init__(self):
        pass

    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0


@pytest.fixture
def mcmc_obj():
    model = DummyModel()
    data = model.evaluate({'a': 0., 'b': 0.})
    param_priors = {'a': ['Uniform', 0., 1.], 'b': ['Uniform', 0., 1.]}
    return MCMCSampler(data, model, param_priors, storage_backend='ram')


@pytest.fixture
def part_initer(mcmc_obj):
    return ParticleInitializer(mcmc_obj, temp_schedule=[0., 1.],
                               mpi_comm=DummyComm())


def test_initialize_particles_from_prior(part_initer):
    particles = part_initer.initialize_particles(num_particles=5,
                                                 measurement_std_dev=1.0)

    particle_param_vals = np.array([p.params.values() for p in particles])
    particle_loglikes = np.array([p.log_like for p in particles])
    particle_weights = np.array([p.weight for p in particles])

    expected_loglikes = np.array([-3. / 2 * np.log(2 * np.pi)] * 5)
    expected_weights = np.exp(expected_loglikes)

    assert all([0. <= val <= 1. for val in particle_param_vals.flatten()])
    np.testing.assert_array_equal(particle_loglikes, expected_loglikes)
    np.testing.assert_array_equal(particle_weights, expected_weights)


def test_set_proposal_distribution_with_scales(part_initer):
    proposal_center = {'a': 1, 'b': 2}
    proposal_scales = {'a': 0.5, 'b': 1}
    part_initer.set_proposal_distribution(proposal_center, proposal_scales)
    assert part_initer.proposal_center == proposal_center
    assert part_initer.proposal_scales == proposal_scales


def test_set_proposal_distribution_with_no_scales(part_initer):
    proposal_center = {'a': 1, 'b': 2}
    part_initer.set_proposal_distribution(proposal_center)
    assert part_initer.proposal_center == proposal_center
    assert part_initer.proposal_scales == {'a': 1, 'b': 1}


def test_initialize_particles_with_proposals(part_initer):
    pass
