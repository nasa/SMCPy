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

from abc import ABC, abstractmethod
from scipy.optimize import bisect
from tqdm import tqdm

from .initializer import Initializer
from .mutator import Mutator
from .updater import Updater
from ..resampler_rngs import *
from ..utils.context_manager import ContextManager
from ..utils.mpi_utils import rank_zero_output_only, rank_zero_run_only
from ..utils.progress_bar import progress_bar
from ..utils.storage import InMemoryStorage


class SamplerBase:
    def __init__(self, mcmc_kernel, show_progress_bar):
        self._mcmc_kernel = mcmc_kernel
        self._initializer = Initializer(self._mcmc_kernel)
        self._mutator = Mutator(self._mcmc_kernel)
        self._updater = None
        self._step_list = []
        self._phi_sequence = []
        self._show_progress_bar = show_progress_bar

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
        """
        Performs SMC sampling. Returns step list and estimates of marginal
        log likelihood at each step.
        """
        raise NotImplementedError

    def _initialize(self, num_particles):
        if self._result and self._result.is_restart:
            self._step = self._result[-1]
            self._phi_sequence = self._result.phi_sequence
            return None
        return self._initializer.initialize_particles(num_particles)

    def _do_smc_step(self, phi, num_mcmc_samples):
        self._mcmc_kernel.path.phi = phi
        particles = self._updater.update(self.step)
        mut_particles = self._mutator.mutate(particles, num_mcmc_samples)
        self.step = mut_particles

    @rank_zero_run_only
    def _save_step(self, step):
        self._result.save_step(step)

    @progress_bar
    def _init_progress_bar(self, phi_idx=-1):
        pbar = False
        bar_format = (
            "{desc}: {percentage:.2f}%|{bar}| "
            + "phi: {n:.5f}/{total_fmt} [{elapsed}<{remaining}"
        )
        pbar = tqdm(total=1.0, bar_format=bar_format)
        pbar.set_description(f"[ mutation ratio: {self.step.attrs['mutation_ratio']}")
        pbar.update(self.phi_sequence[phi_idx])
        return pbar

    @progress_bar
    def _update_progress_bar(self, pbar, dphi):
        pbar.set_description(f"[ mutation ratio: {self.step.attrs['mutation_ratio']}")
        pbar.update(dphi)

    @progress_bar
    def _close_progress_bar(self, pbar):
        pbar.close()


class FixedSampler(SamplerBase):
    """
    SMC sampler using a fixed phi sequence.
    """

    def __init__(self, mcmc_kernel, show_progress_bar=True):
        """
        :param mcmc_kernel: a kernel object for conducting particle mutation
        :type mcmc_kernel: KernelBase object
        """
        super().__init__(mcmc_kernel, show_progress_bar)

    def sample(
        self,
        num_particles,
        num_mcmc_samples,
        phi_sequence,
        ess_threshold,
        resample_rng=standard,
        particles_warn_threshold=0.01,
    ):
        """
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
        """
        self._updater = Updater(
            ess_threshold,
            self._mcmc_kernel,
            resample_rng=resample_rng,
            particles_warn_threshold=particles_warn_threshold,
        )
        self._phi_sequence = phi_sequence

        self.step = self._initialize(num_particles)

        phi_iterator = self._phi_sequence[1:]
        pbar = self._init_progress_bar(0)
        for i, phi in enumerate(phi_iterator):
            self._do_smc_step(phi, num_mcmc_samples)
            self._update_progress_bar(pbar, self._mcmc_kernel.path.delta_phi)
        self._close_progress_bar(pbar)
        return self._result, self._result.estimate_marginal_log_likelihoods()


class AdaptiveSampler(SamplerBase):
    """
    SMC sampler using an adaptive phi sequence.
    """

    def __init__(self, mcmc_kernel, show_progress_bar=True):
        """
        :param mcmc_kernel: a kernel object for conducting particle mutation
        :type mcmc_kernel: KernelBase object
        """
        self.req_phi_index = None
        super().__init__(mcmc_kernel, show_progress_bar)

    def sample(
        self,
        num_particles,
        num_mcmc_samples,
        target_ess=0.8,
        min_dphi=None,
        resample_rng=standard,
        particles_warn_threshold=0.01,
    ):
        """
        :param num_particles: number of particles
        :type num_particles: int
        :param num_mcmc_samples: number of MCMC samples to draw from the
            MCMC kernel per iteration per particle
        :type num_mcmc_samples: int
        :param target_ess: controls adaptive stepping by picking the next
            phi such that the effective sample size is equal to the threshold.
            Specified as a fraction of total number particles (between 0 and 1).
        :type target_ess: float
        :param min_dphi: minimum allowable delta phi for a given SMC step
        :type min_dphi: float
        """
        if target_ess <= 0.0 or target_ess >= 1.0:
            raise ValueError

        self._updater = Updater(
            ess_threshold=1,  # ensures always resampling
            mcmc_kernel=self._mcmc_kernel,
            resample_rng=resample_rng,
            particles_warn_threshold=particles_warn_threshold,
        )
        self._phi_sequence = [0]

        self.step = self._initialize(num_particles)

        pbar = self._init_progress_bar()

        while self._phi_sequence[-1] < 1:
            proposed_phi = self.optimize_step(
                self.step, self._phi_sequence[-1], target_ess
            )
            proposed_phi = self._verify_min_dphi(proposed_phi, min_dphi)
            self._do_smc_step(proposed_phi, num_mcmc_samples)
            self._phi_sequence.append(proposed_phi)
            self._update_progress_bar(pbar, self._mcmc_kernel.path.delta_phi)

        self._close_progress_bar(pbar)

        self.req_phi_index = [
            i
            for i, phi in enumerate(self.phi_sequence)
            if phi in self._mcmc_kernel.path.required_phi_list
        ]

        return self._result, self._result.estimate_marginal_log_likelihoods()

    @rank_zero_output_only
    def optimize_step(self, particles, phi_old, target_ess=1):
        phi = 1
        if not self._full_step_meets_target(phi_old, particles, target_ess):
            phi = bisect(
                self.predict_ess_margin,
                phi_old,
                1,
                args=(phi_old, particles, target_ess),
            )
        proposed_phi_list = self._mcmc_kernel.path.required_phi_list
        proposed_phi_list.append(phi)
        return self._select_phi(proposed_phi_list, phi_old)

    def predict_ess_margin(self, phi_new, phi_old, particles, target_ess):
        delta_phi = phi_new - phi_old
        log_beta = (
            self._get_inc_weights(particles, phi_new)
            if delta_phi > 0
            else np.zeros_like(particles.log_likes)
        )
        numer = 2 * particles._logsum(log_beta)
        denom = particles._logsum(2 * log_beta)
        ESS = 0
        if numer > -np.inf and denom > -np.inf:
            ESS = np.exp(numer - denom)
        return ESS - particles.num_particles * target_ess

    def _get_inc_weights(self, particles, phi_new):
        kernel = self._mcmc_kernel
        kernel.path.phi = phi_new
        args = (
            particles.params,
            particles.log_likes,
            kernel.get_log_priors(particles.param_dict),
        )
        inc_weights = kernel.path.inc_log_weights(*args)
        kernel.path.undo_phi_set()
        return inc_weights

    @staticmethod
    def _select_phi(proposed_phi_list, phi_old):
        return min([p for p in proposed_phi_list if p > phi_old])

    def _verify_min_dphi(self, phi, min_dphi):
        dphi = phi - self._phi_sequence[-1]
        if min_dphi and dphi < min_dphi:
            return self._phi_sequence[-1] + min_dphi
        return phi

    def _full_step_meets_target(self, phi_old, particles, target_ess):
        return self.predict_ess_margin(1, phi_old, particles, target_ess) > 0
