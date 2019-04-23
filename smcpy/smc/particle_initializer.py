'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRessED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNess FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLess THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
'''

from copy import copy
from pymc import Normal
from ..particles.particle import Particle
from ..utils.single_rank_comm import SingleRankComm
import numpy as np
import warnings


class ParticleInitializer():
    '''
    Class to initialize particles prior to Sequential Monte Carlo sampling.
    '''

    def __init__(self, mcmc, temp_schedule, mpi_comm=SingleRankComm()):
        self._mcmc = mcmc
        self.temp_schedule = temp_schedule
        self.proposal_center = None
        self.proposal_scales = None
        self._size = mpi_comm.Get_size()
        self._rank = mpi_comm.Get_rank()

    def initialize_particles(self, measurement_std_dev, num_particles):
        '''
        Initializes first set of particles based on the prior and proposal
        distributions.

        :param measurement_std_dev: standard deviation of the measurement error
        :type measurement_std_dev: float
        :param num_particles: number of particles to use during sampling
        :type num_particles: int
        '''
        self.num_particles = num_particles
        m_std = measurement_std_dev
        self._mcmc.generate_pymc_model(fix_var=True, std_dev0=m_std)
        num_particles_per_partition = self._get_num_particles_per_partition()
        particles = []
        prior_variables = self._create_prior_random_variables()
        if self.proposal_center is not None:
            proposal_variables = self._create_proposal_random_variables()
        else:
            proposal_variables = None
        for _ in range(num_particles_per_partition):
            part = self._create_particle(prior_variables, proposal_variables)
            particles.append(part)
        return particles

    def set_proposal_distribution(self, proposal_center, proposal_scales=None):
        '''
        Given the proposal center and (optionally) the proposal scales, sets
        the proposal distribution for the initial batch of particles

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
        '''
        self._check_proposal_dist_inputs(proposal_center, proposal_scales)
        if proposal_center is not None and proposal_scales is None:
            msg = 'No scales given; setting scales to identity matrix.'
            warnings.warn(msg)
            proposal_scales = {k: 1. for k in self._mcmc.params.keys()}
        if proposal_center is not None and proposal_scales is not None:
            self._check_proposal_dist_input_keys(proposal_center,
                                                 proposal_scales)
            self._check_proposal_dist_input_vals(proposal_center,
                                                 proposal_scales)
        self.proposal_center = proposal_center
        self.proposal_scales = proposal_scales
        return None

    def _get_num_particles_per_partition(self,):
        num_particles_per_partition = self.num_particles / self._size
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
        centers = self.proposal_center
        scales = self.proposal_scales
        random_variables = dict()
        for key in self._mcmc.params.keys():
            variance = (centers[key] * scales[key])**2
            random_variables[key] = Normal(key, centers[key], 1 / variance)
        return random_variables

    def _create_particle(self, prior_variables, prop_variables=None):
        if prop_variables is None:
            params = self._sample_random_variables(prior_variables)
            prior_logp = self._compute_log_prob(prior_variables)
            prop_logp = prior_logp
        else:
            params = self._sample_random_variables(prop_variables)
            prop_logp = self._compute_log_prob(prop_variables)
            self._set_random_variables_value(prior_variables, params)
            prior_logp = self._compute_log_prob(prior_variables)
        log_like = self._evaluate_likelihood(params)
        temp_step = self.temp_schedule[1]
        log_weight = log_like * temp_step + prior_logp - prop_logp
        return Particle(params, log_weight, log_like)

    def _sample_random_variables(self, random_variables):
        param_keys = self._mcmc.params.keys()
        params = {key: random_variables[key].random() for key in param_keys}
        return params

    @staticmethod
    def _set_random_variables_value(random_variables, params):
        for key, value in params.iteritems():
            random_variables[key].value = value
        return None

    @staticmethod
    def _compute_log_prob(random_variables):
        param_log_prob = np.sum([rv.logp for rv in random_variables.values()])
        return param_log_prob

    def _evaluate_likelihood(self, param_vals):
        '''
        Note: this method performs 1 model evaluation per call.
        '''
        mcmc = copy(self._mcmc)
        for key, value in param_vals.iteritems():
            index = mcmc.pymc_mod_order.index(key)
            mcmc.pymc_mod[index].value = value
        results_index = mcmc.pymc_mod_order.index('results')
        results_rv = mcmc.pymc_mod[results_index]
        log_like = results_rv.logp
        return log_like

    @staticmethod
    def _check_proposal_dist_inputs(proposal_center, proposal_scales):
        if not isinstance(proposal_center, (dict, None.__class__)):
            raise TypeError('Proposal center must be a dictionary or None.')
        if not isinstance(proposal_scales, (dict, None.__class__)):
            raise TypeError('Proposal scales must be a dictionary or None.')
        if proposal_center is None and proposal_scales is not None:
            raise ValueError('Proposal scales given but center == None.')
        return None

    @staticmethod
    def _check_proposal_dist_input_vals(proposal_center, proposal_scales):
        center_vals = proposal_center.values()
        scales_vals = proposal_scales.values()
        if not all(isinstance(x, (float, int)) for x in center_vals):
            raise TypeError('"proposal_center" values should be int or float')
        if not all(isinstance(x, (float, int)) for x in scales_vals):
            raise TypeError('"proposal_scales" values should be int or float')
        return None

    def _check_proposal_dist_input_keys(self, proposal_center, proposal_scales):
        if sorted(proposal_center.keys()) != sorted(self._mcmc.params.keys()):
            raise KeyError('"proposal_center" keys != self.parameter_names')
        if sorted(proposal_scales.keys()) != sorted(self._mcmc.params.keys()):
            raise KeyError('"proposal_scales" keys != self.parameter_names')
        return None
