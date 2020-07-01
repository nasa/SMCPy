import numpy as np
import pandas
import pytest

from collections import namedtuple

from smcpy.smc.initializer import Initializer
from smcpy.mcmc.translator_base import Translator


class StubParticles:

    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights


@pytest.fixture
def initializer(stub_mcmc_kernel, stub_comm, mocker):
    mocker.patch('smcpy.smc.initializer.Particles', new=StubParticles)
    initializer = Initializer(stub_mcmc_kernel, phi_init=2, mpi_comm=stub_comm)
    return initializer


@pytest.mark.parametrize('rank,expected', [(0, 4), (1, 3), (2, 3)])
def test_get_num_particles_in_partition(stub_comm, stub_mcmc_kernel, rank,
                                         expected, mocker):
    mocker.patch.object(stub_comm, 'Get_size', return_value=3)
    mocker.patch.object(stub_comm, 'Get_rank', return_value=rank)
    initializer = Initializer(stub_mcmc_kernel, None, mpi_comm=stub_comm)
    assert initializer.get_num_particles_in_partition(10, rank) == expected


def test_mcmc_kernel_not_translator_instance():
    with pytest.raises(TypeError):
        initializer = Initializer(None, None)


def test_initialize_particles_from_prior(initializer, mocker):
    mocker.patch.object(initializer, 'get_num_particles_in_partition',
                        new=lambda x, y: x)
    particles = initializer.init_particles_from_prior(5)

    expected_a_vals = [1, 1, 1, 2, 2]
    expected_b_vals = [2, 2, 2, 3, 3]
    expected_log_like = [0.1, 0.1, 0.1, 0.2, 0.2]
    expected_log_weight = [0.2] * 3 + [0.4] * 2

    np.testing.assert_array_almost_equal(particles.params['a'], expected_a_vals)
    np.testing.assert_array_almost_equal(particles.params['b'], expected_b_vals)
    np.testing.assert_array_almost_equal(particles.log_likes,
                                         expected_log_like)
    np.testing.assert_array_almost_equal(particles.log_weights,
                                         expected_log_weight)


@pytest.mark.parametrize('rank,expected_params,dataframe',
                         [(0, {'a': [3, 3, 3, 4, 4], 'b': [1, 1, 1, 2, 2]}, 0),
                          (1, {'a': [1, 1, 1, 2, 2], 'b': [2, 2, 2, 5, 5]}, 0),
                          (1, {'a': [1, 1, 1, 2, 2], 'b': [2, 2, 2, 5, 5]}, 1),
                          (2, {'a': [3, 4, 5, 5], 'b': [1, 2, 3, 4]}, 0)])
def test_initialize_particles_from_samples(rank, expected_params, initializer,
                                           dataframe, mocker):
    samples = {'a': np.array([3, 3, 3, 4, 4, 1, 1, 1, 2, 2, 3, 4, 5, 5]),
               'b': np.array([1, 1, 1, 2, 2, 2, 2, 2, 5, 5, 1, 2, 3, 4])}
    if dataframe:
        samples = pandas.DataFrame(samples)

    proposal_pdensities = np.array(samples['a']) * 0.1

    initializer._size = 3
    initializer._rank = rank
    
    expected_length = len(expected_params['a'])
    mocker.patch.object(initializer.mcmc_kernel, 'get_log_likelihoods',
            return_value=np.array([0.1, 0.1, 0.1, 0.2, 0.2])[:expected_length])
    mocker.patch.object(initializer.mcmc_kernel, 'get_log_priors',
            return_value=np.array([0.2, 0.2, 0.2, 0.3, 0.3])[:expected_length])

    particles = initializer.init_particles_from_samples(samples,
                                                        proposal_pdensities)

    expected_log_like = [0.1, 0.1, 0.1, 0.2, 0.2][:expected_length]
    expected_log_prior = np.array([0.4] * 3 + [0.7] * 2)[:expected_length]
    expected_log_prop = np.log(np.array(expected_params['a']) * 0.1)
    expected_log_weight = expected_log_prior - expected_log_prop

    np.testing.assert_array_equal(particles.params['a'], expected_params['a'])
    np.testing.assert_array_equal(particles.params['b'], expected_params['b'])
    np.testing.assert_array_equal(particles.log_likes, expected_log_like)
    np.testing.assert_array_equal(particles.log_weights, expected_log_weight)
