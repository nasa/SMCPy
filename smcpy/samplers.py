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

from scipy.optimize import bisect
from tqdm import tqdm

from .sampler_base import SamplerBase
from .smc.updater import Updater
from .utils.progress_bar import set_bar
from .utils.mpi_utils import rank_zero_output_only


class FixedSampler(SamplerBase):

    def __init__(self, mcmc_kernel):
        super().__init__(mcmc_kernel)

    @rank_zero_output_only
    def sample(self, num_particles, num_mcmc_samples, phi_sequence,
               ess_threshold, proposal=None, progress_bar=False):
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
        :param progress_bar: display progress bar during sampling
        :type progress_bar: bool
        '''
        self._updater = Updater(ess_threshold)

        step_list = [self._initialize(num_particles, proposal)]

        phi_iterator = phi_sequence[1:]
        if progress_bar:
            phi_iterator = tqdm(phi_iterator)
        set_bar(phi_iterator, 1, self._mutation_ratio, self._updater)

        for i, phi in enumerate(phi_iterator):
            dphi = phi - phi_sequence[i]
            step_list.append(self._do_smc_step(step_list[-1], phi, dphi,
                                               num_mcmc_samples))
            set_bar(phi_iterator, i + 2, self._mutation_ratio, self._updater)

        return step_list, self._estimate_marginal_log_likelihoods()



class AdaptiveSampler(SamplerBase):

    def __init__(self, mcmc_kernel):
        self.phi_sequence = None
        self.req_phi_index = None
        super().__init__(mcmc_kernel)

    @rank_zero_output_only
    def sample(self, num_particles, num_mcmc_samples, target_ess=1,
               proposal=None, required_phi=1):
        '''
        :param num_particles: number of particles
        :type num_particles: int
        :param num_mcmc_samples: number of MCMC samples to draw from the
            MCMC kernel per iteration per particle
        :type num_mcmc_samples: int
        :param target_ess: controls adaptive stepping by picking the next
            phi such that the effective sample size is equal to the threshold.
            Specified as a fraction of total number particles (between 0 and 1).
        :type target_ess: float
        :param proposal: tuple of samples from a proposal distribution used to
            initialize the SMC sampler; first element is a dictionary with keys
            equal to parameter names and values equal to corresponding samples;
            second element is array of corresponding proposal PDF values
        :type proposal: tuple(dict, array)
        :param required_phi: specific values of phi that must be included in
            the phi sequence (regardless of optimized step)
        :type required_phi: float, int, or list
        '''
        self._updater = Updater(ess_threshold=1) # ensure always resampling

        step_list = [self._initialize(num_particles, proposal)]

        phi_sequence = [0]
        while phi_sequence[-1] < 1:
            phi = self.optimize_step(step_list[-1], phi_sequence[-1],
                                     target_ess, required_phi)
            dphi = phi - phi_sequence[-1]
            step_list.append(self._do_smc_step(step_list[-1], phi, dphi,
                                               num_mcmc_samples))
            phi_sequence.append(phi)

        self.phi_sequence = np.array(phi_sequence)
        self.req_phi_index = [i for i, phi in enumerate(phi_sequence) \
                              if phi in required_phi]

        return step_list, self._estimate_marginal_log_likelihoods()

    def optimize_step(self, particles, phi_old, target_ess=1, required_phi=1):
        phi = bisect(self.predict_ess_margin, phi_old, 1,
                     args=(phi_old, particles, target_ess))
        proposed_phi_list = self._as_phi_list(required_phi)
        proposed_phi_list.append(phi)
        return self._select_phi(proposed_phi_list, phi_old)

    def predict_ess_margin(self, phi_new, phi_old, particles, target_ess=1):
        beta = np.exp((phi_new - phi_old) * particles.log_likes)
        ESS = np.sum(beta) ** 2 / np.sum(beta ** 2)
        return ESS - particles.num_particles * target_ess

    @staticmethod
    def _as_phi_list(phi):
        if not isinstance(phi, list):
            phi = [phi]
        phi = sorted([p for p in phi if p < 1])
        phi.append(1)
        return phi

    @staticmethod
    def _select_phi(proposed_phi_list, phi_old):
        return min([p for p in proposed_phi_list if p > phi_old])
