import numpy as np

from .smc.initializer import Initializer
from .smc.updater import Updater
from .smc.mutator import Mutator

class SMCSampler:

    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel

    def sample(self, num_particles, num_mcmc_samples, phi_sequence,
               ess_threshold):

        initializer = Initializer(self._mcmc_kernel, phi_sequence[1])
        updater = Updater(ess_threshold)
        mutator = Mutator(self._mcmc_kernel)

        particles = initializer.init_particles_from_prior(num_particles)
        step_list = [particles]
        for i, phi in enumerate(phi_sequence[2:]):
            particles = updater.update(particles, phi - phi_sequence[i + 1])
            particles = mutator.mutate(particles, phi, num_mcmc_samples)
            step_list.append(particles)

        return step_list

    @staticmethod
    def estimate_marginal_likelihood(step_list, phi_sequence):
        num_particles = step_list[0].num_particles

        delta_phi = np.tile(np.diff(phi_sequence)[1:], (num_particles, 1))
        log_weights = np.zeros(delta_phi.shape)
        log_likes = np.zeros(delta_phi.shape)

        for i, step in enumerate(step_list[:-1]):
            log_weights[:, i] = step.log_weights.flatten()
            log_likes[:, i] = step.log_likes.flatten()

        Z = np.exp(log_weights + log_likes * delta_phi)

        return np.prod(Z.sum(axis=0))
