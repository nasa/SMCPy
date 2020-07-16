import numpy as np

from tqdm import tqdm

from .smc.initializer import Initializer
from .smc.updater import Updater
from .smc.mutator import Mutator
from .utils.progress_bar import set_bar

class SMCSampler:

    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel

    def sample(self, num_particles, num_mcmc_samples, phi_sequence,
               ess_threshold, progress_bar=False):

        initializer = Initializer(self._mcmc_kernel, phi_sequence[1])
        updater = Updater(ess_threshold)
        mutator = Mutator(self._mcmc_kernel)

        particles = initializer.init_particles_from_prior(num_particles)
        particles = updater.resample_if_needed(particles)
        step_list = [particles]

        phi_iterator = phi_sequence[2:]
        if progress_bar:
            phi_iterator = tqdm(phi_iterator)
        set_bar(phi_iterator, 1, mutation_ratio=0, updater=updater)

        for i, phi in enumerate(phi_iterator):
            particles = updater.update(particles, phi - phi_sequence[i + 1])
            particles = mutator.mutate(particles, phi, num_mcmc_samples)
            step_list.append(particles)

            mutation_ratio = self._compute_mutation_ratio(*step_list[-2:])
            set_bar(phi_iterator, i + 2, mutation_ratio, updater)

        return step_list

    @classmethod
    def estimate_marginal_log_likelihood(clss_, step_list, phi_sequence):
        num_particles = step_list[0].num_particles

        delta_phi = np.diff(phi_sequence)[1:].reshape(1, -1)
        log_weights = np.zeros((num_particles, delta_phi.shape[1]))
        log_likes = np.zeros((num_particles, delta_phi.shape[1]))

        for i, step in enumerate(step_list[:-1]):
            log_weights[:, i] = step.log_weights.flatten()
            log_likes[:, i] = step.log_likes.flatten()

        marg_log_like = log_weights + log_likes * delta_phi
        marg_log_like = clss_._logsum(marg_log_like)

        return np.sum(marg_log_like)

    @staticmethod
    def _logsum(Z):
        Z = -np.sort(-Z, axis=0) # descending
        Z0 = Z[0, :]
        Z_shifted = Z[1:, :] - Z0
        return Z0 + np.log(1 + np.sum(np.exp(Z_shifted), axis=0))

    @staticmethod
    def _compute_mutation_ratio(old_particles, new_particles):
        mutated = ~np.all(new_particles.params == old_particles.params, axis=1)
        return sum(mutated) / new_particles.params.shape[0]
