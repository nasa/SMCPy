import numpy as np
import pytest

from smcpy import SMCSampler


@pytest.fixture
def phi_sequence():
    return np.linspace(0, 1, 11)


@pytest.fixture
def step_list(phi_sequence, mocker):
    num_particles = 5
    step_list = []
    for phi in phi_sequence[1:]:
        particles = mocker.Mock()
        particles.log_weights = np.ones(num_particles).reshape(-1, 1)
        particles.log_likes = np.ones(num_particles).reshape(-1, 1) * phi
        particles.num_particles = num_particles
        step_list.append(particles)
    return step_list


def test_sample(mocker):
    init = mocker.patch('smcpy.smc_sampler.Initializer', autospec=True)
    upd = mocker.patch('smcpy.smc_sampler.Updater', autospec=True)
    mut = mocker.patch('smcpy.smc_sampler.Mutator', autospec=True)

    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    phi_sequence = np.ones(num_steps)

    mcmc_kernel = mocker.Mock()
    ess_threshold = 0.2

    smc = SMCSampler(mocker.Mock())
    step_list = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                           ess_threshold)

    init.assert_called_once_with(smc._mcmc_kernel, phi_sequence[1])
    upd.assert_called_once_with(ess_threshold)
    mut.assert_called_once_with(smc._mcmc_kernel)

    assert len(step_list) == len(phi_sequence) - 1


def test_marginal_likelihood_estimator(step_list, phi_sequence):
    dphi = phi_sequence[1] - phi_sequence[0]
    Z_exp = np.prod([np.sum(np.exp(step.log_weights + step.log_likes * dphi)) \
                     for step in step_list[:-1]])
    Z = SMCSampler.estimate_marginal_likelihood(step_list, phi_sequence)
    assert Z == Z_exp


