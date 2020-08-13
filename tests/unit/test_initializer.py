import numpy as np
import pandas
import pytest

from collections import namedtuple

from smcpy.smc.initializer import Initializer


class StubParticles:

    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights


@pytest.fixture
def initializer(stub_mcmc_kernel, mocker):
    mocker.patch('smcpy.smc.initializer.Particles', new=StubParticles)
    initializer = Initializer(stub_mcmc_kernel)
    return initializer


def test_mcmc_kernel_not_kernel_instance():
    with pytest.raises(TypeError):
        initializer = Initializer(None, None)


def test_initialize_particles_from_prior(initializer, mocker):
    particles = initializer.init_particles_from_prior(5)

    expected_a_vals = [1, 1, 1, 2, 2]
    expected_b_vals = [2, 2, 2, 3, 3]
    expected_log_like = [0.1, 0.1, 0.1, 0.2, 0.2]
    expected_log_weight = [np.log(1 / len(expected_a_vals))] \
                           * len(expected_a_vals)

    np.testing.assert_array_almost_equal(particles.params['a'], expected_a_vals)
    np.testing.assert_array_almost_equal(particles.params['b'], expected_b_vals)
    np.testing.assert_array_almost_equal(particles.log_likes,
                                         expected_log_like)
    np.testing.assert_array_almost_equal(particles.log_weights,
                                         expected_log_weight)


@pytest.mark.parametrize('dataframe', [True, False])
def test_initialize_particles_from_samples(initializer, dataframe, mocker):

    expected_a_params = [3, 3, 3, 4, 4]
    expected_b_params = [1, 1, 1, 2, 2]
    expected_log_like = [0.1, 0.1, 0.1, 0.2, 0.2]
    expected_log_prior = [0.2, 0.2, 0.2, 0.3, 0.3]

    samples = {'a': np.array(expected_a_params),
               'b': np.array(expected_b_params)}

    proposal_pdensities = np.array(samples['a']) * 0.1
    expected_log_weight = expected_log_prior - np.log(proposal_pdensities)

    mocker.patch.object(initializer.mcmc_kernel, 'get_log_likelihoods',
                        return_value=np.array([0.1, 0.1, 0.1, 0.2, 0.2]))
    mocker.patch.object(initializer.mcmc_kernel, 'get_log_priors',
                        return_value=np.array([0.2, 0.2, 0.2, 0.3, 0.3]))

    if dataframe:
        samples = pandas.DataFrame(samples)
    particles = initializer.init_particles_from_samples(samples,
                                                        proposal_pdensities)

    np.testing.assert_array_equal(particles.params['a'], expected_a_params)
    np.testing.assert_array_equal(particles.params['b'], expected_b_params)
    np.testing.assert_array_equal(particles.log_likes, expected_log_like)
    np.testing.assert_array_equal(particles.log_weights, expected_log_weight)
