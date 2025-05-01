"""
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
"""

from .particles import Particles
from ..mcmc.kernel_base import KernelBase
from copy import copy
import numpy as np


class Mutator:
    """
    Mutates particles using an MCMC kernel.
    """

    def __init__(self, mcmc_kernel):
        """
        :param mcmc_kernel: a kernel object for conducting particle mutation
        :type mcmc_kernel: KernelBase object
        """
        self.mcmc_kernel = mcmc_kernel
        self._compute_cov = True  # hidden option to turn off cov (used for objs)

    def mutate(self, particles, num_samples):
        cov = particles.compute_covariance() if self._compute_cov else None
        mutated = self.mcmc_kernel.mutate_particles(
            particles.param_dict, num_samples, cov
        )
        new_particles = Particles(mutated[0], mutated[1], particles.log_weights)
        new_particles.attrs.update(
            {
                "total_unnorm_log_weight": particles.total_unnorm_log_weight,
                "phi": self._mcmc_kernel.path.phi,
                "mutation_ratio": self._compute_mutation_ratio(
                    particles, new_particles
                ),
            }
        )
        return new_particles

    def _compute_mutation_ratio(self, old_particles, new_particles):
        mutated = ~np.all(new_particles.params == old_particles.params, axis=1)
        return sum(mutated) / new_particles.params.shape[0]

    @property
    def mcmc_kernel(self):
        return self._mcmc_kernel

    @mcmc_kernel.setter
    def mcmc_kernel(self, mcmc_kernel):
        if not isinstance(mcmc_kernel, KernelBase):
            raise TypeError
        self._mcmc_kernel = mcmc_kernel
