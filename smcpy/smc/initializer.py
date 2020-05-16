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

from ..mcmc.translator_base import Translator
from ..particles.particle import Particle
from ..smc.smc_step import SMCStep
from ..utils.single_rank_comm import SingleRankComm
import numpy as np

from .mpi_base_class import MPIBaseClass

class Initializer(MPIBaseClass):
    '''
    Generates SMCStep objects based on either samples from a prior distribution
    or given input samples from a sampling distribution.
    '''
    def __init__(self, mcmc_kernel, phi_init, mpi_comm=SingleRankComm()):
        self.phi_init = phi_init
        self.mcmc_kernel = mcmc_kernel
        super().__init__(mpi_comm)

    def initialize_particles_from_prior(self, num_particles):
        '''
        Use model stored in MCMC kernel to sample initial set of particles.

        :param num_particles: number of particles to sample (total across all
            ranks)
        :type num_particles: int
        '''
        n_particles_in_part = self.get_num_particles_in_partition(num_particles,
                                                                  self._rank)
        prior_samples = self.mcmc_kernel.sample_from_prior(n_particles_in_part)
        param_names = list(prior_samples.keys())
        param_values = np.array([prior_samples[pn] for pn in param_names]).T

        particles = []
        for vals in param_values:
            params = dict(zip(param_names, vals))
            log_like = self.mcmc_kernel.get_log_likelihood(params)
            non_norm_log_weight = log_like * self.phi_init
            particles.append(Particle(params, non_norm_log_weight, log_like))

        smc_step = SMCStep()
        smc_step.particles = self._comm.gather(particles, root=0)[0]
        return smc_step

    def initialize_particles_from_samples(self, samples, proposal_pdensity):
        '''
        Initialize a set of particles using pre-sampled parameter values and
        the corresponding prior pdf at those values.

        :param samples: samples of parameters used to initialize particles;
            must be a dictionary with keys = parameter names and values =
            parameter values. Can also be a pandas DataFrame object for this
            reason.
        :type samples: dict, pandas.DataFrame
        :param proposal_pdensity: corresponding probability density function
            values; must be aligned with samples
        :type proposal_pdensity: list or nd.array
        '''
        num_particles = len(proposal_pdensity)
        n_particles_in_parts = \
            [self.get_num_particles_in_partition(num_particles, rank) for \
             rank in range(self._size)]

        param_names = list(samples.keys())
        param_values = np.array([samples[pn] for pn in param_names]).T
        log_proposal_pdensity = np.log(proposal_pdensity)

        start_index = int(np.sum(n_particles_in_parts[:self._rank]))
        end_index = start_index + n_particles_in_parts[self._rank]
        param_values = param_values[start_index:end_index]
        log_proposal_pdensity = log_proposal_pdensity[start_index:end_index]

        particles = []
        for i, vals in enumerate(param_values):
            params = dict(zip(param_names, vals))
            log_like = self.mcmc_kernel.get_log_likelihood(params)
            log_prior = self.mcmc_kernel.get_log_prior(params)
            non_norm_log_weight = log_like * self.phi_init + log_prior - \
                                  log_proposal_pdensity[i]
            particles.append(Particle(params, non_norm_log_weight, log_like))

        smc_step = SMCStep()
        smc_step.particles = self._comm.gather(particles, root=0)[0]
        return smc_step

    @property
    def mcmc_kernel(self):
        return self._mcmc_kernel

    @mcmc_kernel.setter
    def mcmc_kernel(self, mcmc_kernel):
        if not isinstance(mcmc_kernel, Translator):
            raise TypeError
        self._mcmc_kernel = mcmc_kernel
