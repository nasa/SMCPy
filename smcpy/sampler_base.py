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

from abc import ABC, abstractmethod

from .smc.initializer import Initializer
from .smc.mutator import Mutator
from .utils.storage import InMemoryStorage
from .utils.mpi_utils import rank_zero_run_only
from .utils.context_manager import ContextManager


class SamplerBase:

    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel
        self._initializer = Initializer(self._mcmc_kernel)
        self._mutator = Mutator(self._mcmc_kernel)
        self._updater = None
        self._mutation_ratio = 1
        self._step_list = []
        self._phi_sequence = []

        try:
            self._result = ContextManager.get_context()
        except:
            self._result = InMemoryStorage()

    @property
    def phi_sequence(self):
        return np.array(self._phi_sequence)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        if step is not None:
            self._save_step(step)
            self._step = step

    @abstractmethod
    def sample(self):
        '''
        Performs SMC sampling. Returns step list and estimates of marginal
        log likelihood at each step.
        '''
        raise NotImplementedError

    def _initialize(self, num_particles, proposal):
        if self._result and self._result.is_restart:
            self._step = self._result[-1]
            self._phi_sequence = self._result.phi_sequence
            return None
        elif proposal:
            return self._initializer.init_particles_from_samples(*proposal)
        return self._initializer.init_particles_from_prior(num_particles)

    def _do_smc_step(self, phi, delta_phi, num_mcmc_samples):
        particles = self._updater.update(self.step, delta_phi)
        mut_particles = self._mutator.mutate(particles, phi, num_mcmc_samples)
        self._compute_mutation_ratio(particles, mut_particles)
        self.step = mut_particles

    def _compute_mutation_ratio(self, old_particles, new_particles):
        mutated = ~np.all(new_particles.params == old_particles.params, axis=1)
        self._mutation_ratio = sum(mutated) / new_particles.params.shape[0]

    @rank_zero_run_only
    def _save_step(self, step):
        self._result.save_step(step)
