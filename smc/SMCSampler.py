from copy import copy
import itertools
from mpi4py import MPI
from mcmc import MCMC
import numpy as np
import os
import pickle
from pymc import Normal
from smc import Particle
from smc import ParticleChain
import warnings

# for multiprocessing pickle fix
from smc.pickle_methods import _pickle_method, _unpickle_method
import types
import copy_reg

class SMCSampler(object):

    def __init__(self, data, model, param_priors):
        self.comm, self.size, self.rank = self._set_up_communicator()
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
    

    def sample(self, num_particles, num_time_steps, measurement_std_dev,
               ESS_threshold=None, proposal_center=None, proposal_scales=None,
               restart_time=0):
        '''
        :param num_particles: number of particles to use during sampling
        :type num_particles: int
        :param num_time_steps: number of time steps in temperature schedule that
            is used to transition between prior and posterior distributions.
        :type num_time_steps: int
        :param std_dev: standard deviation of the measurement error
        :type std_dev: float
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
        '''
        self._set_num_particles(num_particles)
        self._set_ESS_threshold(ESS_threshold)
        self._set_temperature_schedule(num_time_steps)
        if restart_time == 0:
            self._set_proposal_distribution(proposal_center, proposal_scales)
            self._set_start_time_based_on_proposal()
            particles = self._initialize_particles(measurement_std_dev)
        elif 0 < restart_time <= num_time_steps:
            #TODO allow restart; load particle chain from hdf5
            #particles = self._load_particle_chain_from_hdf5()
            self._set_start_time_equal_to_restart_time(restart_time)
        else:
            raise ValueError('restart_time not in range [0, num_time_steps]')
        for t in range(num_time_steps)[self.start_time_step:]:
            temperature_step = self.temp_schedule[t] - self.temp_schedule[t-1]
            if self.rank == 0:
                particles = self.comm.gather(particles, root=0)
                # TODO: NEED TO GATHER ALL PARTICLES AND STORE IN CHAIN...
                # CAN THIS BE DONE HERE SO THAT IT DOESN'T HAVE TO BE IN INIT?
        return None


    def _set_num_particles(self, num_particles):
        self.num_particles = num_particles
        return None


    def _set_ESS_threshold(self, ESS_threshold):
        if ESS_threshold is None:
            ESS_threshold = 0.25*num_particles
        self.ESS_threshold = ESS_threshold
        return None


    def _set_temperature_schedule(self, num_cooling_steps):
        #TODO add option to optimize temp_schedule
        self.temp_schedule = np.linspace(0, 1, num_cooling_steps)
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


    def _set_start_time_equal_to_restart_time(self, restart_time):
        self.start_time_step = restart_time
        return None


    @staticmethod
    def compute_particle_weights(particle_set, temperature_step):
        new_particles = particle_set.copy()
        for p in new_particles:
            p.weight = np.log(p.weight)+p.log_like*temperature_step
        particle_set.add_step(new_particles)
        particle_set.normalize_step_weights()
        return particle_set


############## ############## ##############


def run_smc4mpi(data, model, param_priors, data_std, q_opt, M, N, T,
                ESS_threshold=None, proposal_scales=None, save=True,
                save_every=False, save_filename = 'smc.p', comm=None,
                restart_t=0, load_filename=None):
    '''
    :param data: array-like object containing observations that will
        be compared to the output of model (i.e., must be same shape as
        model output).
    :type array: ndarray
    :param model: A model object with an evaluate() method that
        returns output in the same shape as the data. Should follow the
        standard SHRMPy format.
    :type model: class instance
    :param param_priors: A dictionary containing prior information in
        SHRMPy format (same as MCMC class). Keys are parameter names,
        and values are lists containing distribution info; e.g.,
        {'a': ['Uniform', lower_bound, upper_bound], 'b': ['Normal', mean,
        variance]}
    :type param_priors: dict
    :param data_std: estimated standard deviation of measurement error
    :type data_std: float
    :param M: number of particles to use
    :type M: int
    :param N: number of MCMC steps in particle transition
    :type N: int
    :param T: number of steps in the cooling sequence
    :type T: int
    :param ESS_threshold: the effective sample size threshold;
        defaults to 0.25*M, or a quarter of the number of particles.
    :type ESS_threshold: float or int
    :param q_opt (optional): A dictionary containing optimized
        parameters (e.g., from an initial least squares fit). Keys
        correspond to parameter names given in params. For example:
        {'a': 1.0, 'b': 2.0}. Used to create a proposal distribution
        centered at q_opt with covariance defined by proposal_scales.
    :type q_opt: dict
    :param proposal_scales: A dictionary in the same form as q_opt,
        except the values correspond scales which will be used to scale the
        covariance of the proposal distribution centered at q_opt. This
        parameter defaults to a dictionary of ones; e.g., {'a': 1, 'b': 1}.
        The proposal is thus ~N(q_opt, I*q_opt*scales). This parameter is
        only used if q_opt is provided.  
    :type proposal_scales: dict
    :param save: determines whether or not to save particle chain to pickle
        at the end of the analysis
    :type save: Boolean
    :param save_every: determines whether or not to save particle chain to
        pickle after every update of the cooling sequence
    :type save_every: Boolean
    :param save_filename: filename of pickle into which particle chain is dumped
    :type save_filename: string
    :param restart_t: the time index at which to restart the SMC sampler (this
        should be a time step for which the pchain has already been evaluated
        (i.e., weights computed, etc.). The restart will then being the process
        of obtaining particles for target t+1 and so on.
    :type restart_t: int
    :param load_filename: the pickle file containing the stored pchain object
        to use in the restart process (only used if a restart_t is specified)
    :type restart_t: string
    '''
    
    # NO RESTART, INITIALIZE
    if restart_t == 0:
        # initialize particles
        # TODO: better way of sending each proc the length of its partition
        #       when partition hasn't been made. Right now gen dummy list of
        #       particles.
        part_particles = np.array_split(range(M), size)
        part_particles = comm.scatter(part_particles, root=0)
        particles = initialize_particles(len(part_particles),
                                         cooling_sequence[0],
                                         param_priors, data_std, copy(mcmc),
                                         proposal_scales, q_opt)

        new_particles = comm.gather(particles, root=0)

        # initialize particle chain and partition it on root
        if my_rank == 0: 
            # concatenate params_list (which is a list of lists at this point)
            new_particles = sum(new_particles, [])

            # set up particle chain
            print 'initiating chain...'
            pchain = ParticleChain(M)
            pchain.add_step(new_particles)
            pchain.normalize_step_weights()
        
            # resample if needed
            print 'checking for initial resample...'
            pchain = resample_if_needed(pchain, ESS_threshold)

    # YES RESTART, LOAD
    elif restart_t > 0 and restart_t <= T:
        # make sure restart file is given
        if load_filename is None:
            raise NameError('if restarting, need to specify load_filename')
        # get pchain on rank 0
        if my_rank == 0:
            # load particle system
            print os.getcwd()
            with open(load_filename, 'r') as pf:
                print 'loading pickle file for restart: %s' % load_filename
                pchain = pickle.load(pf)
                print 'pickle file loaded.'

            # rebuild pchain if starting from earlier than most recent step
            if pchain.nsteps > restart_t:
                print 'rebuilding chain...'
                new_pchain = ParticleChain(M)
                for i in range(restart_t+1):
                    new_pchain.add_step(pchain.step[i])
                pchain = new_pchain
            print 'reloaded pchain, ready to resume.'

    else:
        ins = (restart_t, T)
        raise ValueError('Restart step greater than total steps, %s > %s' % ins)


    # BEGIN SEQUENCE
    for i in range(T)[restart_t+1:]:
        print '%s percent complete...' % (i/T*100.)
        # compute weights of given last step's likelihoods and weights
        cooling_step = cooling_sequence[i]-cooling_sequence[i-1]
        if my_rank == 0:
            pchain = compute_weights(pchain, cooling_step)
            pchain = resample_if_needed(pchain, ESS_threshold)
            cov = pchain.calculate_step_covariance()
            
            # check to make sure cov is positive definite
            try:
                np.linalg.cholesky(cov)
            except np.linalg.linalg.LinAlgError:
                print 'estimated chain covariance is not pos def, setting = I'
                cov = np.eye(cov.shape[0])

            # partition particle chain
            part_particles = np.array_split(pchain.step[-1], size)

        else:
            cov = None
            part_particles = [None]

        # scatter cov for all to use
        # NOTE: there's prob a more official way of sending this value to all
        cov = comm.scatter([cov]*size, root=0)

        # scatter particles
        part_particles = comm.scatter(part_particles, root=0)

        # mutate partitioned particle list
        new_particles = []
        for j, p in enumerate(part_particles):
            new_particles.append(mutate(copy(mcmc), N, cov, cooling_sequence[i],
                                        data_std, p))
            ins = (my_rank, j)
        part_particles = new_particles

        # reassemble particle chain
        part_particles = comm.gather(part_particles, root=0)
        if my_rank == 0:
            pchain.overwrite_step(-1, np.concatenate(part_particles))
            if save_every is True:
                # save particle chain after t^th gather if save_every
                pchain.save(save_filename)
                # write restart info to text file
                with open('restart.txt', 'w') as rf:
                    ins = (i, cooling_sequence[i])
                    rf.write('saved pchain for t=%s and phi_t=%s' % ins)

    if my_rank == 0:
        # save
        if save == True:
            pchain.save(save_filename)
        return my_rank, pchain
    else:
        return my_rank, None


    
def initialize_particles(M_part, cooling_step, param_priors, data_std,
                         mcmc, proposal_scales=None, q_opt=None):
    '''
    Initializes particle chain by sampling from priors and computing weights.
    '''
    # sample params
    params_list = []
    for m in range(M_part):
        params = dict()
        if q_opt is None:
            mcmc.generate_pymc_model(fix_var=True, std_dev0=data_std)
            pymc_mod_order = mcmc.pymc_mod_order
            pymc_mod = mcmc.pymc_mod
            for key in param_priors.keys():
                index = pymc_mod_order.index(key)
                params[key] = pymc_mod[index].random()
            params['pymc_mod_order'] = pymc_mod_order
            params['pymc_mod'] = pymc_mod
        else:
            #TODO really need a better way of doing this + bounding
            if m == 0:
                sd =[abs(q_opt[k])*proposal_scales[k] for k in q_opt.keys()]
                cov = np.eye(len(q_opt))*np.array(sd)**2
                rv = multi_norm(q_opt.values(), cov)
            rvs = rv.rvs()
            params = {q_opt.keys()[i]: r for i, r in enumerate(rvs)}
            # set up pymc model at those params
            mcmc.generate_pymc_model(fix_var=True, std_dev0=data_std, q0=params)
            params['pymc_mod_order'] = mcmc.pymc_mod_order
            params['pymc_mod'] = mcmc.pymc_mod
            params['log_prob'] = rv.logpdf(rvs)

        params_list.append(params)

    # generate particles
    new_particles = []
    for p in params_list:
        # get prior probabilities
        pymc_mod = params['pymc_mod']
        pymc_mod_order = params['pymc_mod_order']
        prior_log_prob = []
        for key, value in p.iteritems():
            if key not in ['log_prob', 'pymc_mod', 'pymc_mod_order']:
                index = pymc_mod_order.index(key)
                pymc_mod[index].value = value
                prior_log_prob.append(pymc_mod[index].logp)
        prior_log_prob = np.sum(prior_log_prob)

        # get proposal probabilities
        if 'log_prob' in p.keys():
            prop_log_prob = p['log_prob']
            del p['log_prob']
        else:
            prop_log_prob = prior_log_prob

        # evaluate model, compute likelihood
        log_like = pymc_mod[pymc_mod_order.index('results')].logp

        # compute weight given likelihood and prior probability
        #TODO allow for importance sampling from distribution other than prior
        log_weight = log_like*cooling_step+prior_log_prob-prop_log_prob

        # generate particle
        del p['pymc_mod']
        del p['pymc_mod_order']
        new_particles.append(Particle(p, log_weight, log_like))

    return new_particles


def resample_if_needed(pchain, ESS_threshold):
    '''
    Resample if ESS below threshold
    '''
    ESS = pchain.compute_ESS()
    if ESS < ESS_threshold:
        #if self.verbose > 0:
        #    if self.verbose > 1:
        print 'ESS = %s' % ESS
        print 'resampling...'
        pchain.resample(overwrite=True)
    else:
        #if self.verbose > 2:
        print 'ESS = %s' % ESS
        print 'no resampling required.'
    return pchain


def compute_weights(pchain, cooling_step):
    '''
    Computes weights based on updating cooling step.
    '''
    # copy old particles
    new_particles = pchain.copy()

    # update weights
    for p in new_particles:
        p.weight = np.log(p.weight)+p.log_like*cooling_step
    pchain.add_step(new_particles)
    pchain.normalize_step_weights()
    return pchain


# TODO: make this operate on a list of particles (ie include for)
def mutate(mcmc, N, cov, cooling_step, data_std, particle):
    '''
    Mutates a single particle via MCMC kernel. Returns a new particle with
    updated params and log likelihood. Retains weights.
    '''
    # NOTE does this need to be reset like this?
    mcmc.generate_pymc_model(fix_var=True, std_dev0=data_std,
                             q0 = particle.params)

    # set initial values to particle params
    #for key, value in particle.params.iteritems():
    #    index = mcmc.pymc_mod_order.index(key)
    #    mcmc.pymc_mod[index].value = value

    # sample
    mcmc.sample(N, burnin=0, step_method='smc_metropolis', cov=cov,
                verbose=True, phi=cooling_step)

    # generate new particle
    stochastics = mcmc.db.getstate()['stochastics']
    particle.params = {key: stochastics[key] for key in particle.params.keys()}
    particle.log_like = mcmc.MCMC.logp

    return particle
