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

from copy import copy

from .particles import Particles
from ..mcmc.kernel_base import MCMCKernel
from ..utils.single_rank_comm import SingleRankComm


class Initializer:
    '''
    Generates SMCStep objects based on either samples from a prior distribution
    or given input samples from a sampling distribution.
    '''
    def __init__(self, mcmc_kernel):
        self.mcmc_kernel = mcmc_kernel

    @property
    def mcmc_kernel(self):
        return self._mcmc_kernel

    @mcmc_kernel.setter
    def mcmc_kernel(self, mcmc_kernel):
        if not isinstance(mcmc_kernel, MCMCKernel):
            raise TypeError
        self._mcmc_kernel = mcmc_kernel

    def __call__(self, samples):
        '''
        Initialize a set of particles using pre-sampled parameter values and
        the corresponding prior pdf at those values.

        :param samples: samples of parameters used to initialize particles;
            must be a dictionary with keys = parameter names and values =
            parameter values. Can also be a pandas DataFrame object for this
            reason.
        :type samples: dict, pandas.DataFrame
        '''
        log_likes = self.mcmc_kernel.get_log_likelihoods(samples)
        log_priors = self.mcmc_kernel.get_log_priors(samples)
        log_weights = np.full(log_likes.shape, np.log(1 / log_likes.shape[1]))
        particles = Particles(samples, log_likes, log_weights)
        return particles
