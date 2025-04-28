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

import numpy as np

from copy import copy

from .particles import Particles
from ..mcmc.kernel_base import KernelBase
from ..utils.single_rank_comm import SingleRankComm


class Initializer:
    """
    Generates SMCStep objects based on either samples from a prior distribution
    or a proposal distribution, depending on how the target path is defined in
    the MCMC kernel (i.e., if a proposal is available, it will be used).
    """

    def __init__(self, mcmc_kernel):
        """
        :param mcmc_kernel: a kernel object for conducting particle mutation
        :type mcmc_kernel: KernelBase object
        """
        self.mcmc_kernel = mcmc_kernel

    @property
    def mcmc_kernel(self):
        return self._mcmc_kernel

    @mcmc_kernel.setter
    def mcmc_kernel(self, mcmc_kernel):
        if not isinstance(mcmc_kernel, KernelBase):
            raise TypeError
        self._mcmc_kernel = mcmc_kernel

    def initialize_particles(self, num_particles):
        kernel = self.mcmc_kernel
        if kernel.has_proposal():
            params = kernel.sample_from_proposal(num_particles)
        else:
            params = kernel.sample_from_prior(num_particles)

        log_likes = kernel.get_log_likelihoods(params)
        log_weights = np.array([np.log(1 / num_particles)] * num_particles)

        particles = Particles(params, log_likes, log_weights)
        particles.attrs.update({"phi": 0, "mutation_ratio": 0})
        return particles
