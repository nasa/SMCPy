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
import copy
import warnings
from smcpy.particles.particle import Particle
from smcpy.utils.checks import Checks


def _mpi_decorator(func):
    def wrapper(self, *args, **kwargs):
        if self._rank == 0:
            func(self, *args, **kwargs)
    return wrapper


class SMCStep(Checks):
    """ A single step of the sequential monte carlo (SMC) method that contains
    a list of Particle instances

    :param particles: list of particle instances
    :type particles: list
    """

    def __init__(self):
        self.particles = []

    def add_particle(self, particle):
        '''
        Add a single particle to the step.

        :param particle: single instance of an SMC particle
        :type particle: Particle class object
        '''
        self.particles.append(self._check_particle(particle))

    def set_particles(self, particles):
        '''
        Fill a list of particles in the step with ID.

        :param particles: list of particle instances
        :type particles: list
        '''
        self.particles = self._check_step(particles)
        return None

    def copy(self):
        '''
        Returns a copy of the entire step class.
        '''
        return copy.deepcopy(self)

    def get_likes(self):
        '''
        Returns a list of likelihoods for each particle in the step
        '''
        return [np.exp(p.log_like) for p in self.particles]

    def get_log_likes(self):
        '''
        Returns a list of log(likelihoods) for each particle in the step
        '''
        return [p.log_like for p in self.particles]

    def get_mean(self):
        '''
        Returns the mean of each parameter within the step
        '''
        normalized_weights = self.normalize_step_weights()
        param_names = self.particles[0].params.keys()
        mean = {}
        for pn in param_names:
            mean[pn] = []
            for i, p in enumerate(self.particles):
                mean[pn].append(normalized_weights[i] * p.params[pn])
            mean[pn] = np.sum(mean[pn])
        return mean

    def get_log_weights(self):
        '''
        Returns a list of the log weights of each particle in the step
        '''
        return [p.log_weight for p in self.particles]

    def calculate_covariance(self):
        '''
        Estimates the covariance matrix for the step.
        '''
        particle_list = self.particles
        normalized_weights = self.normalize_step_weights()
        means = np.array(self.get_mean().values())

        cov_list = []
        for i, p in enumerate(particle_list):
            param_vector = p.params.values()
            diff = (param_vector - means).reshape(-1, 1)
            R = np.dot(diff, diff.transpose())
            cov_list.append(normalized_weights[i] * R)
        cov_matrix = np.sum(cov_list, axis=0)
        cov_matrix = cov_matrix * (float(len(cov_list)) / (len(cov_list) - 1))

        if not self._is_positive_definite(cov_matrix):
            msg = 'current step cov not pos def, setting to identity matrix'
            warnings.warn(msg)
            cov_matrix = np.eye(cov_matrix.shape[0])

        return cov_matrix

    def normalize_step_log_weights(self):
        '''
        Normalizes log weights, and then transforms back into to log space for
        all particles inside the step
        '''
        normalized_weights = self.normalize_step_weights()
        for index, p in enumerate(self.particles):
            p.log_weight = np.log(normalized_weights[index])
        return None

    def normalize_step_weights(self):
        '''
        Normalizes log weights of all particles inside the step
        '''
        log_weights = np.array(self.get_log_weights())
        shifted_weights = np.exp(log_weights - max(log_weights))
        normalized_weights = shifted_weights / sum(shifted_weights)
        return normalized_weights

    def compute_ess(self):
        '''
        Computes the effective sample size (ess) of the step based on log weight
        '''
        self.normalize_step_log_weights()
        log_weights = self.get_log_weights()
        return 1 / np.sum([np.exp(w)**2 for w in log_weights])

    def get_params(self, key):
        '''
        Retrieves parameter values in every particle of a specific parameter

        :param key: parameter name
        :type key: str
        '''
        particles = self.particles
        return np.array([p.params[key] for p in particles])

    def get_param_dicts(self):
        '''
        Retrieves the entire parameter dictionary for every particle
        '''
        particles = self.particles
        return [p.params for p in particles]

    def get_particles(self):
        '''
        Retrieves the list of particles within the step object
        '''
        return self.particles

    def resample(self):  # issue here
        '''
        Resamples the step based on normalized weights. Assigns discrete
        probabilities to each particle (sum to 1), resample from this discrete distribution using the particle's copy() method.
        '''
        particles = self.particles
        num_particles = len(particles)
        weights = np.exp(self.get_log_weights())
        weights_cs = np.cumsum(weights)
        # intervals based on weights to use for discrete probability draw
        intervals = zip(np.insert(weights_cs, 0, 0)[:-1], weights_cs)
        # generate random numbers, iterate to find intervals for resample
        R = np.random.uniform(0, 1, [num_particles, ])
        new_particles = []
        uniform_weight = 1. / num_particles
        for r in R:
            for i, (a, b) in enumerate(intervals):

                if a <= r < b:
                    # resample
                    new_particles.append(particles[i].copy())
                    # assign uniform weight
                    new_particles[-1].log_weight = uniform_weight
                    break
        self.particles = new_particles
        return None

    def print_particle_info(self, particle_num):
        """
        Prints the particle number and its information

        :param int particle_num: index of the desired particle
        """
        particle = self.particles[particle_num]
        print '-----------------------------------------------------'
        print 'Particle: %s' % particle_num
        particle.print_particle_info()
        return None

    @_mpi_decorator
    def plot_marginal(self, key, save=False, show=True,
                      prefix='marginal_'):  # pragma no cover
        '''
        Plots a single marginal approximation for param given by <key>.
        '''
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for p in self.particles:
            ax.plot([p.params[key], p.params[key]], [0.0, np.exp(p.log_weight)])
            ax.plot(p.params[key], np.exp(p.log_weight), 'o')
        if save:
            plt.savefig(prefix + key + '.png')
        if show:
            plt.show()
        plt.close(fig)
        return None

    @_mpi_decorator
    def plot_pairwise_weights(self, param_names=None, labels=None,
                              save=False, show=True, param_lims=None,
                              label_size=None, tick_size=None, nbins=None,
                              prefix='pairwise'):  # pragma no cover
        '''
        Plots pairwise distributions of all parameter combos. Color codes each
        by weight.
        '''
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        # get particles
        particles = self.particles

        # set up label dictionary
        if param_names is None:
            param_names = particles[0].params.keys()
        if labels is None:
            labels = param_names
        label_dict = {key: lab for key, lab in zip(param_names, labels)}
        if param_lims is not None:
            lim_dict = {key: l for key, l in zip(param_names, param_lims)}
        if nbins is not None:
            bin_dict = {key: n for key, n in zip(param_names, nbins)}
        L = len(param_names)

        # setup figure
        fig = plt.figure(figsize=[10 * (L - 1) / 2, 10 * (L - 1) / 2])

        # create lower triangle to obtain param combos
        tril = np.tril(np.arange(L**2).reshape([L, L]), -1)
        ikeys = np.transpose(np.nonzero(tril)).tolist()

        # use lower triangle to id subplots
        tril = np.tril(np.arange((L - 1)**2).reshape([L - 1, L - 1]) + 1)
        iplts = [i for i in tril.flatten() if i > 0]

        log_weights = self.get_log_weights()
        means = self.get_mean()
        for i in zip(iplts, ikeys):
            iplt = i[0]     # subplot index
            ikey1 = i[1][1]  # key index for xparam
            ikey2 = i[1][0]  # key index for yparam
            key1 = param_names[ikey1]
            key2 = param_names[ikey2]
            ax = {key1 + '+' + key2: fig.add_subplot(L - 1, L - 1, iplt)}
            # get list of all particle params for key1, key2 combinations
            pkey1 = []
            pkey2 = []
            for p in particles:
                pkey1.append(p.params[key1])
                pkey2.append(p.params[key2])
            # plot parameter combos with weight as color

            def rnd_to_sig(x):
                return round(x, -int(np.floor(np.log10(abs(x)))) + 1)
            sc = ax[key1 + '+' + key2].scatter(pkey1, pkey2, c=log_weights, vmin=0.0,
                                               vmax=rnd_to_sig(max(log_weights)))
            ax[key1 + '+' + key2].axvline(means[key1], color='C1', linestyle='--')
            ax[key1 + '+' + key2].axhline(means[key2], color='C1', linestyle='--')
            ax[key1 + '+' + key2].set_xlabel(label_dict[key1])
            ax[key1 + '+' + key2].set_ylabel(label_dict[key2])

            # if provided, set x y lims
            if param_lims is not None:
                ax[key1 + '+' + key2].set_xlim(lim_dict[key1])
                ax[key1 + '+' + key2].set_ylim(lim_dict[key2])
            # if provided set font sizes
            if tick_size is not None:
                ax[key1 + '+' + key2].tick_params(labelsize=tick_size)
            if label_size is not None:
                ax[key1 + '+' + key2].xaxis.label.set_size(label_size)
                ax[key1 + '+' + key2].yaxis.label.set_size(label_size)
            # if provided, set x ticks
            if nbins is not None:
                ax[key1 + '+' + key2].locator_params(axis='x', nbins=bin_dict[key1])
                ax[key1 + '+' + key2].locator_params(axis='y', nbins=bin_dict[key2])

        fig.tight_layout()

        # colorbar
        if L <= 2:
            cb = plt.colorbar(sc, ax=ax[key1 + '+' + key2])
        else:
            ax1_position = fig.axes[0].get_position()
            ax3_position = fig.axes[2].get_position()
            y0 = ax1_position.y0
            x0 = ax3_position.x0
            w = 0.02
            h = abs(ax1_position.y1 - ax1_position.y0)
            empty_ax = fig.add_axes([x0, y0, w, h])
            cb = plt.colorbar(sc, cax=empty_ax)
            if tick_size is not None:
                empty_ax.tick_params(labelsize=tick_size)

        cb.ax.get_yaxis().labelpad = 15
        cb.ax.set_ylabel('Normalized weights', rotation=270)

        plt.tight_layout()

        if save:
            plt.savefig(prefix + '.png')
        if show:
            plt.show()
        plt.close(fig)
        return None

    def _check_step(self, particle_list):
        if not isinstance(particle_list, (list, np.ndarray)):
            raise TypeError('Input must be a list or numpy array')
        for particle in particle_list:
            self._check_particle(particle)
        return particle_list

    @staticmethod
    def _check_particle(particle):
        if not isinstance(particle, Particle):
            raise TypeError('Input must be a of the Particle class')
        return particle
