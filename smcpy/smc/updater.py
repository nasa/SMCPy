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

from .particles import Particles


class Updater:
    '''
    Updates particle weights (and resamples if necessary) based on delta phi
    and particle state.
    '''
    def __init__(self, ess_threshold):
        '''
        :param ess_threshold: threshold on effective sample size (ess); if
            ess < ess_threshold, resampling with replacement is conducted.
        :type ess_threshold: float
        '''
        self.ess_threshold = ess_threshold
        self._ess = np.nan
        self._unnorm_log_weights = []

    @property
    def ess_threshold(self):
        return self._ess_threshold

    @ess_threshold.setter
    def ess_threshold(self, ess_threshold):
        if ess_threshold < 0. or ess_threshold > 1.:
            raise ValueError('ESS threshold must be between 0 and 1.')
        self._ess_threshold = ess_threshold

    @property
    def ess(self):
        return self._ess

    @property
    def resampled(self):
        return self._resampled

    @nvtx.annotate(color='darkslategray')
    def __call__(self, particles, delta_phi):
        new_log_weights = self._compute_new_weights(particles, delta_phi)
        with nvtx.annotate(color='darkslategray', message='create Particles'):
            new_particles = Particles(particles.param_dict, particles.log_likes,
                                      new_log_weights)

        new_particles = self.resample_if_needed(new_particles)
        return new_particles

    @nvtx.annotate(color='darkslategray')
    def resample_if_needed(self, new_particles):
        eff_sample_size = new_particles.compute_ess()
        self._ess = eff_sample_size
        self._resampled = False
        resample_mask = eff_sample_size < self.ess_threshold * \
                                          new_particles.num_particles
        return self._resample(new_particles, resample_mask)

    @nvtx.annotate(color='darkslategray')
    def _compute_new_weights(self, particles, delta_phi):
        un_log_weights = particles.log_weights + particles.log_likes * delta_phi
        self._unnorm_log_weights.append(un_log_weights)
        return un_log_weights

    @nvtx.annotate(color='darkslategray')
    def _resample(self, particles, resample_mask):
        with nvtx.annotate(message='initialize arrays', color='darkred'):
            new_params = particles.params.copy()
            new_log_likes = particles.log_likes.copy()

        with nvtx.annotate(message='sample and bin', color='darkred'):
            u_samples = np.random.uniform(0, 1, particles.num_particles)
            bins = np.cumsum(particles.weights, axis=1)

        with nvtx.annotate(message='loop over models', color='darkred'):
            for i, (u, b) in enumerate(zip(u_samples, bins)):
                if not resample_mask[i]:
                    continue
                resample_indices = np.digitize(u.squeeze(), b.squeeze())
                new_params[i, :, :] = particles.params[i, resample_indices, :]
                new_log_likes[i, :, :] = particles.log_likes[i, resample_indices, :]

        with nvtx.annotate(message='conv to dict', color='darkred'):
            param_dict = dict(zip(particles.param_names,
                              np.transpose(new_params, (2, 0, 1))))

        with nvtx.annotate(message='compute weights', color='darkred'):
            uniform_weight = 1 / particles.num_particles
            new_weights = np.log(np.full(particles.weights.shape, uniform_weight))

        with nvtx.annotate(message='instance Particles', color='darkred'):
            particles = Particles(param_dict, new_log_likes, new_weights)
        return particles
