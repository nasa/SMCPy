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
    def sample(self, num_particles, num_mcmc_samples, phi_sequence,
               ess_threshold, progress_bar=False):

        initializer = Initializer(self._mcmc_kernel)
        updater = Updater(ess_threshold)
        mutator = Mutator(self._mcmc_kernel)

        particles = initializer.init_particles_from_prior(num_particles)
        particles = updater.resample_if_needed(particles)
        step_list = [particles]

        phi_iterator = phi_sequence[1:]
        if progress_bar:
            phi_iterator = tqdm(phi_iterator)
        set_bar(phi_iterator, 1, mutation_ratio=0, updater=updater)

        for i, phi in enumerate(phi_iterator):
            particles = updater.update(particles, phi - phi_sequence[i])
            particles = mutator.mutate(particles, phi, num_mcmc_samples)
            step_list.append(particles)

            mutation_ratio = self._compute_mutation_ratio(*step_list[-2:])
            set_bar(phi_iterator, i + 2, mutation_ratio, updater)

        return step_list, self._estimate_marginal_log_likelihood(updater)

    def _estimate_marginal_log_likelihood(self, updater):
        summed_un_log_wts = [self._logsum(ulw) \
                             for ulw in updater._unnorm_log_weights]
        return np.sum(summed_un_log_wts)

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
