from copy import copy
import numpy as np
import pytest

from smcpy.smc.mutator import Mutator
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel


def test_mcmc_kernel_not_kernel_instance():
    with pytest.raises(TypeError):
        initializer = Mutator(None)


def test_mutate(mocker):
    particles = mocker.Mock()
    particles_mock = mocker.patch('smcpy.smc.mutator.Particles')
    mutator = Mutator(mcmc_kernel=mocker.Mock(VectorMCMCKernel))
    mutator.mcmc_kernel.mutate_particles.return_value = (1, 2, 3)

    _ = mutator(particles, phi=0.5, num_samples=5)

    particles.compute_covariance.assert_called_once()
    particles_mock.called_once_with(1, 2, 3)
