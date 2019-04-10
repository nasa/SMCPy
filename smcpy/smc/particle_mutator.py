from ..utils.single_rank_comm import SingleRankComm
from copy import copy
import numpy as np


class ParticleMutator():

    def __init__(self, step, mcmc, num_mcmc_steps, mpi_comm=SingleRankComm()):
        self.step = step
        self._comm = mpi_comm
        self._mcmc = mcmc
        self.num_mcmc_steps = num_mcmc_steps
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    def mutate_new_particles(self, covariance, measurement_std_dev,
                             temperature_step):
        '''
        Predicts next distribution along the temperature schedule path using
        the MCMC kernel.
        '''
        particles = self._partition_and_scatter_particles()
        mcmc = copy(self._mcmc)
        step_method = 'smc_metropolis'
        new_particles = []
        acceptance_count = 0
        for particle in particles:
            mcmc.generate_pymc_model(fix_var=True, std_dev0=measurement_std_dev,
                                     q0=particle.params)
            mcmc.sample(self.num_mcmc_steps, burnin=0, step_method=step_method,
                        cov=covariance, verbose=-1, phi=temperature_step)
            stochastics = mcmc.MCMC.db.getstate()['stochastics']
            params = {key: stochastics[key] for key in particle.params.keys()}
            if particle.params != params:
                acceptance_count += 1
            particle.params = params
            particle.log_like = mcmc.MCMC.logp
            new_particles.append(particle)

        new_particles = self._gather_and_concat_particles(new_particles)
        self._acceptance_ratio = float(acceptance_count) / len(particles)
        self.step = self._update_step_with_new_particles(new_particles)
        return self.step

    def _update_step_with_new_particles(self, particles):
        if self._rank == 0:
            self.step.set_particles(particles)
        else:
            self.step = None
        return self.step

    def _partition_new_particles(self):
        partitions = np.array_split(self.step.get_particles(),
                                    self._size)
        return partitions

    def _partition_and_scatter_particles(self):
        if self._rank == 0:
            particles = self._partition_new_particles()
        else:
            particles = []
        particles = self._comm.scatter(particles, root=0)
        return particles

    def _gather_and_concat_particles(self, new_particles):
        new_particles = self._comm.gather(new_particles, root=0)

        if self._rank == 0:
            new_particles = list(np.concatenate(new_particles))

        return new_particles
