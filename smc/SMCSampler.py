from copy import copy
import itertools
from mpi4py import MPI
from mcmc import MCMC
import numpy as np
import os
import pickle
from pymc import Normal
from ..particles.particle import Particle
from ..particles.particle_chain import ParticleChain
import warnings

#TODO remove if not needed
## for multiprocessing pickle fix
#from smc.pickle_methods import _pickle_method, _unpickle_method
#import types
#import copy_reg

class SMCSampler(object):

    def __init__(self, data, model, param_priors):
        self.comm, self.size, self.rank = self._setup_communicator()
        self.mcmc = self._setup_mcmc_sampler(data, model, param_priors)


    def _setup_communicator(self,):
        if comm is None:
            comm = MPI.COMM_WORLD
        size = comm.Get_size()
        my_rank = comm.Get_rank()
        return comm, size, my_rank


    def _setup_mcmc_sampler(self, data, model, param_priors):
        mcmc = MCMC(data=data, model=model, params=param_priors,
                    storage_backend='ram')
        return mcmc
    

    def sample(self, num_particles, num_time_steps, num_mcmc_steps,
               measurement_std_dev, ESS_threshold=None, proposal_center=None,
               proposal_scales=None, restart_time=0, hdf5_file_path=None):
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
        :param restart_time: time step at which to restart sampling; default is
            zero, meaning the sampling process starts at the prior distribution;
            note that restart_time < num_time_steps.
        :type restart_time: int
        :param hdf5_file_path: file path of a particle chain saved using the
            ParticleSet.save() method.
        :type hdf5_file_path: string


        :Returns: A ParticleChain class instance that stores all particles and
            their past generations at every time step.
        '''
        self._set_num_particles(num_particles)
        self._set_temperature_schedule(num_time_steps)
        self._set_num_mcmc_steps(num_mcmc_steps)
        self._set_ESS_threshold(ESS_threshold)

        # TODO: move this block to an initialize particle chain function?
        if restart_time == 0:
            self._set_proposal_distribution(proposal_center, proposal_scales)
            self._set_start_time_based_on_proposal()
            particles = self._initialize_particles(measurement_std_dev)
            particle_chain = self.initialize_particle_chain(particles)
        elif 0 < restart_time <= num_time_steps:
            self._set_start_time_equal_to_restart_time(restart_time)
            particle_chain = self._load_particle_chain_from_hdf5(hdf5_file_path)
        else:
            raise ValueError('restart_time not in range [0, num_time_steps]')

        self._set_particle_chain(particle_chain)
        for t in range(num_time_steps)[self.start_time_step:]:
            temperature_step = self.temp_schedule[t] - self.temp_schedule[t-1]
            if self.rank == 0:
                step_cov = self._compute_current_step_covariance()
                self._create_new_particles()
                self._compute_new_particle_weights(temperature_step)
                self._normalize_new_particle_weights()
                self._resample_if_needed()
                partitioned_particles = self._partition_new_particles()
            else:
                step_cov = None
                partitioned_particles = [None]
            #TODO better (faster) way to get covariance to every rank?
            step_cov = self.comm.scatter([step_covariance]*self.size, root=0)
            new_particles = self.comm.scatter(partitioned_particles, root=0)
            new_particles = self._mutate_new_particles(new_particles, step_cov,
                                                       measurement_std_dev,
                                                       temperature_step)
            new_particles = self.comm.gather(particles, root=0)
            self._update_particle_chain_with_new_particles(new_particles)
            self._save_particle_chain_progress(hdf5_file_path)
        return self.particle_chain


    def _set_num_particles(self, num_particles):
        self.num_particles = num_particles
        return None


    def _set_temperature_schedule(self, num_cooling_steps):
        #TODO add option to optimize temp_schedule
        self.temp_schedule = np.linspace(0, 1, num_cooling_steps)
        return None


    def _set_num_mcmc_steps(self, num_mcmc_steps):
        self.num_mcmc_steps = num_mcmc_steps
        return None


    def _set_ESS_threshold(self, ESS_threshold):
        if ESS_threshold is None:
            ESS_threshold = 0.25*num_particles
        self.ESS_threshold = ESS_threshold
        return None


    def _set_proposal_distribution(self, proposal_center, proposal_scales):
        if proposal_center is not None and proposal_scales is None:
            msg = 'No scales given; setting scales to identity matrix.')
            warnings.warn(msg)
            proposal_scales = {k: 1. for k in mcmc.param_priors.keys()}
        elif proposal_center is None and proposal_scales is not None:
            raise ValueError('Proposal scales given but center == None.')
        self.proposal_center = proposal_center
        self.proposal_scales = proposal_scales
        return None


    def _set_start_time_based_on_proposal(self,):
        '''
        If proposal distribution is equal to prior distribution, can start
        Sequential Monte Carlo sampling at time = 1, since prior can be
        sampled directly. If using a different proposal, must first start by
        estimating the prior (i.e., time = 0). This is a result of the way
        the temperature schedule is defined.
        '''
        if self.proposal_center is None:
            self.start_time_step = 1
        else self.proposal_center is not None:
            self.start_time_step = 0
        return None


    def _initialize_particles(self, measurement_std_dev):
        #TODO allow for sampling of data_stddev
        #TODO allow for sampled proposal_center
        m_std = measurement_std_dev
        self.mcmc.generate_pymc_model(fix_var=True, std_dev0=m_std)
        num_particles_per_partition = self._get_num_particles_per_partition()
        particles = []
        if self.proposal_center is None:
            random_variables = self._create_prior_random_variables()
        else:
            random_variables = self._create_proposal_random_variables()
        for i in range(num_particles_per_partition):
            particles.append(self._sample_particle(random_variables))
        return particles


    def _get_num_particles_per_partition(self,):
        num_particles_per_partition = self.num_particles/self.size
        remainder = self.num_particles % self.size
        overtime_ranks = range(remainder):
        if self.rank in overtime_ranks:
            num_particles_per_partition += 1
        return num_particles_per_partition


    def _create_prior_random_variables(self,):
        mcmc = copy(self.mcmc) # TODO necessary?
        random_variables = dict()
        for key in mcmc.params.keys():
            index = mcmc.pymc_mod_order.index(key)
            random_variables[key] = mcmc.pymc_mod[index]
        return random_variables


    def _create_proposal_random_variables(self,):
        centers = self.proposal_centers
        scales = self.proposal_scales
        random_variables = dict()
        for key in self.mcmc.pymc_order:
            variance = (centers[key] * scales[key])**2
            random_variables[key] = Normal(key, centers[key], 1/variance)
        return random_variables


    def _sample_particle(self, random_variables):
        #TODO rename to something specific to init?
        param_keys = self.mcmc.keys()
        param_vals = {key: random_variables[key].random() for key in param_keys}
        prior_log_prob = np.sum([rv.logp for rv in random_variables])
        log_like = self._evaluate_likelihood(param_vals)
        log_weight = log_like*cooling_step + prior_log_prob - prop_log_prob
        return Particle(param_vals, log_weight, log_like)


    def _evaluate_likelihood(self, random_variables, param_vals):
        '''
        Note: this method performs 1 model evaluation per call.
        '''
        mcmc = copy(self.mcmc) # TODO necessary?
        for key, value in param_vals:
            index = mcmc.pymc_mod_order.index(key)
            mcmc.pymc_mod[index].value = value
        results_index = mcmc.pymc_mod_order.index('results')
        log_like = mcmc.pymc_mod[results_index].logp
        return log_like


    def initialize_particle_chain(self, particles):
        particles = self.comm.gather(particles, root=0)
        if self.rank == 0:
            particle_chain = ParticleSet(self.num_particles)
            particle_chain.add_step(particles)
            particle_chain.normalize_step_weights()
        else:
            particle_chain = None
        return particle_chain


    def load_particle_chain_from_hdf5(self, hdf5_file_path):
        '''
        :param hdf5_file_path: file path of a particle chain saved using the
            ParticleSet.save() method.
        :type hdf5_file_path: string
        '''
        if self.rank == 0:
            #load
            return particle_chain
        else:
            return None


    def _set_particle_chain(self, particle_chain):
        self.particle_chain = particle_chain
        return None


    def _set_start_time_equal_to_restart_time(self, restart_time):
        self.start_time_step = restart_time
        return None


    def _create_new_particles(self):
        new_particles = self.particle_chain.copy_step(step=-1)
        self.particle_chain.add_step(new_particles)
        return None


    def _compute_new_particle_weights(self, temperature_step):
        new_particles = self.particle_chain.step[-1]
        for p in new_particles:
            p.weight = np.log(p.weight)+p.log_like*temperature_step
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
            #TODO remove prints or put in verbose flag
            print 'ESS = %s' % ESS
            print 'resampling...'
            self.particle_chain.resample(overwrite=True)
        else:
            print 'ESS = %s' % ESS
            print 'no resampling required.'
        return None


    def _compute_current_step_covariance(self):
        covariance = self.particle_chain.calculate_step_covariance(step=-1)
        if not self.is_positive_definite(covariance):
            msg = 'current step cov not pos def, setting to identity matrix'
            warnings.warn(msg)
            covariance = np.eye(covariance.shape[0])
        return covariance


    @staticmethod
    def is_positive_definite(covariance):
        try:
            np.linalg.cholesky(covariance)
            return True
        except np.linalg.linalg.LinAlgError:
            return False


    def _partition_new_particles(self):
        partitions = np.array_split(self.particle_chain.step[-1], self.size)
        return partitions


    def _mutate_new_particles(particles, covariance, measurement_std_dev):
        '''
        Predicts next distribution along the temperature schedule path using
        the MCMC kernel.
        '''
        mcmc = copy(self.mcmc)
        step_method = 'smc_metropolis'
        new_particles = []
        for i, particle in enumerate(particles):
            mcmc.generate_pymc_model(fix_var=True, std_dev0=measurement_std_dev,
                                     q0=particle.params)
            mcmc.sample(self.num_mcmc_steps, burnin=0, step_method=step_method,
                        cov=covariance, verbose=True, phi=temperature_step)
            stochastics = mcmc.db.getstate()['stochastics']
            params = {key: stochastics[key] for key in particle.params.keys()}
            particle.params = params
            particle.log_like = mcmc.MCMC.logp
            new_particles.append(particle)
        return new_particles


    def _update_particle_chain_with_new_particles(self, new_particles):
        if self.rank == 0:
            particles = np.concatenate(new_particles)
            self.particle_chain.overwrite_step(step=-1, particles)
        return None


    def _save_particle_chain_progress(self, hdf5_file_path):
        #TODO improve save in particle_chain.py; update to save hdf5; save
        #     current temperature schedule and time step as well
        self.particle_chain.save(hdf5_file_path)
        return None
