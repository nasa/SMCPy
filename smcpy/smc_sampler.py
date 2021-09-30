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

import numpy as np
import nvtx

from tqdm import tqdm

from .smc.initializer import Initializer
from .smc.updater import Updater
from .smc.mutator import Mutator
from .utils.mpi_utils import rank_zero_output_only



class SMCSampler:

    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel

    @rank_zero_output_only
    def sample(self, num_particles, num_mcmc_samples, phi_sequence,
               ess_threshold, proposal):
        '''
        :param num_particles: number of particles
        :type num_particles: int
        :param num_mcmc_samples: number of MCMC samples to draw from the
            MCMC kernel per iteration per particle
        :type num_mcmc_samples: int
        :param phi_sequence: increasing monotonic sequence of floats starting
            at 0 and ending at 1; sometimes referred to as tempering schedule
        :type phi_sequence: list or array
        :param ess_threshold: the effective sample size at which resampling
            should be conducted; given as a fraction of num_particles and must
            be in the range [0, 1]
        :type ess_threshold: float
        :param proposal: tuple of samples from a proposal distribution used to
            initialize the SMC sampler; first element is a dictionary with keys
            equal to parameter names and values equal to corresponding samples;
            second element is array of corresponding proposal PDF values
        :type proposal: tuple(dict, array)
        '''
        with nvtx.annotate(message='init helper classes', color='orange'):
            initializer = Initializer(self._mcmc_kernel)
            updater = Updater(ess_threshold)
            mutator = Mutator(self._mcmc_kernel)

        with nvtx.annotate(message='init particles', color='orange'):
            particles = self._initialize(initializer, num_particles, proposal)
            particles = updater.resample_if_needed(particles)

        step_list = [particles]

        phi_iterator = phi_sequence[1:]

        for i, phi in enumerate(phi_iterator):
            with nvtx.annotate(message='update particles', color='orange'):
                particles = updater(step_list[-1], phi - phi_sequence[i])
            with nvtx.annotate(message='mutate particles', color='orange'):
                mut_particles = mutator(particles, phi, num_mcmc_samples)
            with nvtx.annotate(message='append step', color='orange'):
                step_list.append(mut_particles)

            with nvtx.annotate(message='mut. ratio', color='orange'):
                mutation_ratio = self._compute_mutation_ratio(particles,
                                                              mut_particles)

        return step_list, self.estimate_marginal_log_likelihoods(updater)

    def _initialize(self, initializer, num_particles, proposal):
        proposal = self._mcmc_kernel.conv_param_array_to_dict(proposal)
        particles = initializer(proposal)
        return particles

    @nvtx.annotate(message="estimate_mll", color="orange")
    def estimate_marginal_log_likelihoods(self, updater):
        sum_un_log_wts = [self._logsum(ulw) \
                          for ulw in updater._unnorm_log_weights]
        sum_un_log_wts.insert(0, np.zeros(sum_un_log_wts[0].shape))
        return np.cumsum(np.array(sum_un_log_wts).squeeze(), axis=0).T

    @staticmethod
    def _logsum(Z):
        shift = Z.max(axis=1, keepdims=True)
        Z_shifted = Z - shift
        return shift + np.log(np.sum(np.exp(Z_shifted), axis=1, keepdims=True))

    @staticmethod
    def _compute_mutation_ratio(old_particles, new_particles):
        mutated = ~np.all(new_particles.params == old_particles.params, axis=2)
        return np.sum(mutated, axis=1) / new_particles.params.shape[1]
