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

from tqdm import tqdm

from .smc.initializer import Initializer
from .smc.updater import Updater
from .smc.mutator import Mutator
from .utils.progress_bar import set_bar
from .utils.mpi_utils import rank_zero_output_only



class SMCSampler:

    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel

    @rank_zero_output_only
    def sample(self, num_particles, num_mcmc_samples,
               ess_threshold, proposal=None, progress_bar=False,
               normalization_phi=1.0):
        '''
        :param num_particles: number of particles
        :type num_particles: int
        :param num_mcmc_samples: number of MCMC samples to draw from the
            MCMC kernel per iteration per particle
        :type num_mcmc_samples: int
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
        # HACK
        self._ess_threshold = ess_threshold

        # HACK for Bayes factor normalization
        self._normalization_phi = normalization_phi
        self._norm_added = False

        initializer = Initializer(self._mcmc_kernel)
        updater = Updater(ess_threshold)
        mutator = Mutator(self._mcmc_kernel)

        particles = self._initialize(initializer, num_particles, proposal)
        #particles = updater.resample_if_needed(particles)

        step_list = [particles]
        phi_sequence = [0]

        while phi_sequence[-1] < 1:
            phi = self._optimize_step(particles, phi_sequence[-1])
            particles = updater.update(step_list[-1], phi - phi_sequence[-1])
            mut_particles = mutator.mutate(particles, phi, num_mcmc_samples)
            step_list.append(mut_particles)
            phi_sequence.append(phi)

            mutation_ratio = self._compute_mutation_ratio(particles,
                                                          mut_particles)

        # HACK for Bayes factor normalization
        self.phi_sequence = np.array(phi_sequence)
        self.norm_phi_idx = [i for i, phi in enumerate(phi_sequence) \
                             if phi == self._normalization_phi][0]

        return step_list, self._estimate_marginal_log_likelihoods(updater)

    def _initialize(self, initializer, num_particles, proposal):
        if proposal is None:
            particles = initializer.init_particles_from_prior(num_particles)
        else:
            particles = initializer.init_particles_from_samples(*proposal)
        return particles

    def _estimate_marginal_log_likelihoods(self, updater):
        sum_un_log_wts = [self._logsum(ulw) \
                          for ulw in updater._unnorm_log_weights]
        num_updates = len(sum_un_log_wts)
        return [0] + [np.sum(sum_un_log_wts[:i+1]) for i in range(num_updates)]

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

    def _optimize_step(self, particles, phi_old):
        from scipy.optimize import bisect
        # HACKY HACKZ
        self._phi_old = phi_old
        self._temp_particles = particles
        ESS_1 = self._compute_ess(1)
        if ESS_1 > 0:
            return 1
        else:
            phi = bisect(self._compute_ess, phi_old, 1)
            # HACK for Bayes factor normalization
            if phi > self._normalization_phi and not self._norm_added:
                self._norm_added = True
                return self._normalization_phi
            return phi

    def _compute_ess(self, phi):
        phi_old = self._phi_old
        particles = self._temp_particles
        beta = np.exp((phi - phi_old) * particles.log_likes)
        ESS = np.sum(beta) ** 2 / np.sum(beta ** 2)
        return ESS - particles.num_particles * self._ess_threshold

