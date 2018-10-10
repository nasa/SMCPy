import numpy as np
import os
import warnings

from copy import copy
from mpi4py import MPI
from mcmc.MCMC import MCMC
from pymc import Normal
from ..particles.particle import Particle
from ..particles.particle_chain import ParticleChain
from ..hdf5.hdf5_storage import HDF5Storage


class SMCSampler(object):
    '''
    Class for performing parallel Sequential Monte Carlo sampling
    '''

    def __init__(self, data, model, param_priors):
        self._comm, self._size, self._rank = self._setup_communicator()
        self._mcmc = self._setup_mcmc_sampler(data, model, param_priors)


    @staticmethod
    def _setup_communicator():
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        my_rank = comm.Get_rank()
        return comm, size, my_rank


    @staticmethod
    def _setup_mcmc_sampler(data, model, param_priors):
        mcmc = MCMC(data=data, model=model, params=param_priors,
                    storage_backend='ram')
        return mcmc


    def sample(self, num_particles, num_time_steps, num_mcmc_steps,
               measurement_std_dev, ESS_threshold=None, proposal_center=None,
               proposal_scales=None, restart_time_step=0, hdf5_to_load=None,
               autosave_file=None):
        '''
        :param num_particles: number of particles to use during sampling
        :type num_particles: int
        :param num_time_steps: number of time steps in temperature schedule that
            is used to transition between prior and posterior distributions.
        :type num_time_steps: int
        :param num_mcmc_steps: number of mcmc steps to take during mutation
        :param num_mcmc_steps: int
        :param measurement_std_dev: standard deviation of the measurement error
        :type measurement_std_dev: float
        :param ESS_threshold: threshold equivalent sample size; triggers
            resampling when ESS > ESS_threshold
        :type ESS_threshold: float or int
        :param proposal_center: initial parameter dictionary, which is used to
            define the initial proposal distribution when generating particles;
            default is None, and initial proposal distribution = prior.
        :type proposal_center: dict
        :param proposal_scales: defines the scale of the initial proposal
            distribution, which is centered at proposal_center, the initial
            parameters; i.e. prop ~ MultivarN(q1, (I*proposal_center*scales)^2).
            Proposal scales should be passed as a dictionary with keys and
            values corresponding to parameter names and their associated scales,
            respectively. The default is None, which sets initial proposal
            distribution = prior.
        :type proposal_scales: dict
        :param restart_time_step: time step at which to restart sampling;
            default is zero, meaning the sampling process starts at the prior
            distribution; note that restart_time_step < num_time_steps. The
            step at restart_time is retained, and the sampling begins at the
            next step (t=restart_time_step+1).
        :type restart_time_step: int
        :param hdf5_to_load: file path of a particle chain saved using the
            ParticleChain.save() method.
        :type hdf5_to_load: string


        :Returns: A ParticleChain class instance that stores all particles and
            their past generations at every time step.
        '''
        self._set_num_particles(num_particles)
        self._set_temperature_schedule(num_time_steps)
        self._set_num_mcmc_steps(num_mcmc_steps)
        self._set_ESS_threshold(ESS_threshold)
        self._set_autosave_behavior(autosave_file)

        if restart_time_step == 0:
            self._set_proposal_distribution(proposal_center, proposal_scales)
            self._set_start_time_based_on_proposal()
            particles = self._initialize_particles(measurement_std_dev)
            particle_chain = self._initialize_particle_chain(particles)
        elif 0 < restart_time_step <= num_time_steps:
            self._set_start_time_equal_to_restart_time_step(restart_time_step)
            particle_chain = self.load_particle_chain(hdf5_to_load)
            particle_chain = self._trim_particle_chain(particle_chain,
                                                       restart_time_step)
        else:
            raise ValueError('restart time outside range [0, num_time_steps]')

        self._set_particle_chain(particle_chain)
        self._autosave_particle_chain()
        for t in range(num_time_steps)[self._start_time_step+1:]:
            temperature_step = self.temp_schedule[t] - self.temp_schedule[t-1]
            new_particles = self._create_new_particles(temperature_step)
            covariance = self._compute_current_step_covariance()
            mutated_particles = self._mutate_new_particles(new_particles,
                                                           covariance,
                                                           measurement_std_dev,
                                                           temperature_step)
            self._update_particle_chain_with_new_particles(mutated_particles)
            self._autosave_particle_step()
        self._close_autosaver()
        return self.particle_chain


    def _set_num_particles(self, num_particles):
        self.num_particles = num_particles
        return None


    def _set_temperature_schedule(self, num_cooling_steps):
        self.temp_schedule = np.linspace(0, 1, num_cooling_steps)
        return None


    def _set_num_mcmc_steps(self, num_mcmc_steps):
        self.num_mcmc_steps = num_mcmc_steps
        return None


    def _set_ESS_threshold(self, ESS_threshold):
        if ESS_threshold is None:
            ESS_threshold = 0.5*self.num_particles
        self.ESS_threshold = ESS_threshold
        return None


    def _set_autosave_behavior(self, autosave_file):
        self._autosave_file = autosave_file
        if self._autosave_file is not None and self._rank == 0:
            self._autosaver = HDF5Storage(autosave_file, mode='w')
        return None


    def _set_proposal_distribution(self, proposal_center, proposal_scales):
        if proposal_center is not None and proposal_scales is None:
            msg = 'No scales given; setting scales to identity matrix.'
            warnings.warn(msg)
            proposal_scales = {k: 1. for k in self._mcmc.params.keys()}
        elif proposal_center is None and proposal_scales is not None:
            raise ValueError('Proposal scales given but center == None.')
        self._proposal_center = proposal_center
        self._proposal_scales = proposal_scales
        return None


    def _set_start_time_based_on_proposal(self,):
        '''
        If proposal distribution is equal to prior distribution, can start
        Sequential Monte Carlo sampling at time = 1, since prior can be
        sampled directly. If using a different proposal, must first start by
        estimating the prior (i.e., time = 0). This is a result of the way
        the temperature schedule is defined.
        '''
        if self._proposal_center is None:
            self._start_time_step = 1
        else:
            self._start_time_step = 0
        return None


    def _initialize_particles(self, measurement_std_dev):
        m_std = measurement_std_dev
        self._mcmc.generate_pymc_model(fix_var=True, std_dev0=m_std)
        num_particles_per_partition = self._get_num_particles_per_partition()
        particles = []
        prior_variables = self._create_prior_random_variables()
        if self._proposal_center is not None:
            proposal_variables = self._create_proposal_random_variables()
        else:
            proposal_variables = None
        for _ in range(num_particles_per_partition):
            part = self._create_particle(prior_variables, proposal_variables)
            particles.append(part)
        return particles


    def _get_num_particles_per_partition(self,):
        num_particles_per_partition = self.num_particles/self._size
        remainder = self.num_particles % self._size
        overtime_ranks = range(remainder)
        if self._rank in overtime_ranks:
            num_particles_per_partition += 1
        return num_particles_per_partition


    def _create_prior_random_variables(self,):
        mcmc = copy(self._mcmc)
        random_variables = dict()
        for key in mcmc.params.keys():
            index = mcmc.pymc_mod_order.index(key)
            random_variables[key] = mcmc.pymc_mod[index]
        return random_variables


    def _create_proposal_random_variables(self,):
        centers = self._proposal_center
        scales = self._proposal_scales
        random_variables = dict()
        for key in self._mcmc.params.keys():
            variance = (centers[key] * scales[key])**2
            random_variables[key] = Normal(key, centers[key], 1/variance)
        return random_variables


    def _create_particle(self, prior_variables, prop_variables=None):
        param_vals, prior_logp = self._draw_random_variables(prior_variables)
        if prop_variables is None:
            prop_logp = prior_logp
        else:
            param_vals, prop_logp = self._draw_random_variables(prop_variables)
        log_like = self._evaluate_likelihood(param_vals)
        temp_step = self.temp_schedule[self._start_time_step]
        log_weight = log_like*temp_step + prior_logp - prop_logp
        return Particle(param_vals, np.exp(log_weight), log_like)


    def _draw_random_variables(self, random_variables):
        param_keys = self._mcmc.params.keys()
        param_vals = {key: random_variables[key].random() for key in param_keys}
        param_log_prob = np.sum([rv.logp for rv in random_variables.values()])
        return param_vals, param_log_prob


    def _evaluate_likelihood(self, param_vals):
        '''
        Note: this method performs 1 model evaluation per call.
        '''
        mcmc = copy(self._mcmc)
        for key, value in param_vals.iteritems():
            index = mcmc.pymc_mod_order.index(key)
            mcmc.pymc_mod[index].value = value
        results_index = mcmc.pymc_mod_order.index('results')
        log_like = mcmc.pymc_mod[results_index].logp
        return log_like


    def _initialize_particle_chain(self, particles):
        particles = self._comm.gather(particles, root=0)
        if self._rank == 0:
            particle_chain = ParticleChain()
            if self._start_time_step == 1:
                particle_chain.add_step([]) # empty 0th step
            particle_chain.add_step(np.concatenate(particles))
            particle_chain.normalize_step_weights()
        else:
            particle_chain = None
        return particle_chain


    def _set_particle_chain(self, particle_chain):
        self.particle_chain = particle_chain
        return None


    def _set_start_time_equal_to_restart_time_step(self, restart_time_step):
        self._start_time_step = restart_time_step
        return None


    def _trim_particle_chain(self, particle_chain, restart_time_step):
        if self._rank == 0:
            to_keep = range(0, restart_time_step + 1)
            trimmed_steps = [particle_chain.get_particles(i) for i in to_keep]
            particle_chain._steps = trimmed_steps
        return particle_chain


    @staticmethod
    def _file_exists(hdf5_file):
        return os.path.exists(hdf5_file)


    def _compute_current_step_covariance(self):
        if self._rank == 0:
            covariance = self.particle_chain.calculate_step_covariance(step=-1)
            if not self._is_positive_definite(covariance):
                msg = 'current step cov not pos def, setting to identity matrix'
                warnings.warn(msg)
                covariance = np.eye(covariance.shape[0])
        else:
            covariance = None
        covariance = self._comm.scatter([covariance]*self._size, root=0)
        return covariance


    @staticmethod
    def _is_positive_definite(covariance):
        try:
            np.linalg.cholesky(covariance)
            return True
        except np.linalg.linalg.LinAlgError:
            return False


    def _create_new_particles(self, temperature_step):
        if self._rank == 0:
            self._initialize_new_particles()
            self._compute_new_particle_weights(temperature_step)
            self._normalize_new_particle_weights()
            self._resample_if_needed()
            new_particles = self._partition_new_particles()
        else:
            new_particles = [None]
        new_particles = self._comm.scatter(new_particles, root=0)
        return new_particles


    def _initialize_new_particles(self):
        new_particles = self.particle_chain.copy_step(step=-1)
        self.particle_chain.add_step(new_particles)
        return None


    def _compute_new_particle_weights(self, temperature_step):
        for p in self.particle_chain.get_particles(-1):
            p.weight = np.exp(np.log(p.weight)+p.log_like*temperature_step)
        return None


    def _normalize_new_particle_weights(self):
        self.particle_chain.normalize_step_weights()
        return None


    def _resample_if_needed(self):
        '''
        Checks if ESS below threshold; if yes, resample with replacement.
        '''
        ESS = self.particle_chain.compute_ESS()
        if ESS < self.ESS_threshold:
            print 'ESS = %s' % ESS
            print 'resampling...'
            self.particle_chain.resample(overwrite=True)
        else:
            print 'ESS = %s' % ESS
            print 'no resampling required.'
        return None


    def _partition_new_particles(self):
        partitions = np.array_split(self.particle_chain.get_particles(-1),
                                    self._size)
        return partitions


    def _mutate_new_particles(self, particles, covariance, measurement_std_dev,
                              temperature_step):
        '''
        Predicts next distribution along the temperature schedule path using
        the MCMC kernel.
        '''
        mcmc = copy(self._mcmc)
        step_method = 'smc_metropolis'
        new_particles = []
        for particle in particles:
            mcmc.generate_pymc_model(fix_var=True, std_dev0=measurement_std_dev,
                                     q0=particle.params)
            mcmc.sample(self.num_mcmc_steps, burnin=0, step_method=step_method,
                        cov=covariance, verbose=-1, phi=temperature_step)
            stochastics = mcmc.MCMC.db.getstate()['stochastics']
            params = {key: stochastics[key] for key in particle.params.keys()}
            particle.params = params
            particle.log_like = mcmc.MCMC.logp
            new_particles.append(particle)
        new_particles = self._comm.gather(new_particles, root=0)
        return new_particles


    def _update_particle_chain_with_new_particles(self, new_particles):
        if self._rank == 0:
            particles = np.concatenate(new_particles)
            self.particle_chain.overwrite_step(step=-1, particle_list=particles)
        return None


    def _autosave_particle_chain(self):
        if self._rank == 0 and self._autosave_file is not None:
            self._autosaver.write_chain(self.particle_chain)
        return None


    def _autosave_particle_step(self):
        if self._rank == 0 and self._autosave_file is not None:
            step_index = self.particle_chain.get_num_steps() - 1
            step = self.particle_chain.get_particles(step_index)
            self._autosaver.write_step(step, step_index)
        return None


    def _close_autosaver(self):
        if self._rank == 0 and self._autosave_file is not None:
            self._autosaver.close()
        return None


    def save_particle_chain(self, h5_file):
        '''
        Saves self.particle_chain to an hdf5 file using the HDF5Storage class.

        :param hdf5_to_load: file path at which to save particle chain
        :type hdf5_to_load: string
        '''
        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='w')
            hdf5.write_chain(self.particle_chain)
            hdf5.close()
        return None


    def load_particle_chain(self, h5_file):
        '''
        Loads and returns a particle chain object stored using the HDF5Storage
        class.

        :param hdf5_to_load: file path of a particle chain saved using the
            ParticleChain.save() or self.save_particle_chain() methods.
        :type hdf5_to_load: string
        '''
        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='r')
            particle_chain = hdf5.read_chain()
            hdf5.close()
            print 'Particle chain loaded from %s.' % h5_file
        else:
            particle_chain = None
        return particle_chain
