import numpy as np
import copy
from smcpy.particles.particle import Particle


class SMCStep():

    def __init__(self,):
        self.particles = []

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
        return None

    def add_particle(self, particle):
        '''
        Add a single particle to the step.
        '''
        self.particles.append(particle)

    def fill_step(self, particle_list):
        '''
        Fill a list of particles in the step.
        '''
        self.particles = self._check_step(particle_list)
        return None

    def copy_step(self):
        '''
        Returns a copy of particle chain's particle list.
        '''
        return copy.deepcopy(self.particles)

    def copy(self):
        '''
        Returns a copy of the entire step class.
        '''
        return copy.deepcopy(self)

    def get_likes(self):
        return [np.exp(p.log_like) for p in self.particles]

    def get_log_likes(self):
        return [p.log_like for p in self.particles]

    def get_mean(self):
        param_names = self.particles[0].params.keys()
        mean = {}
        for pn in param_names:
            mean[pn] = []
            for p in self.particles:
                mean[pn].append(p.weight * p.params[pn])
            mean[pn] = np.sum(mean[pn])
        return mean

    def get_weights(self):
        return [p.weight for p in self.particles]

    def calculate_covariance(self):
        particle_list = self.particles

        means = np.array(self.get_mean().values())

        cov_list = []
        for p in particle_list:
            param_vector = p.params.values()
            diff = (param_vector - means).reshape(-1, 1)
            R = np.dot(diff, diff.transpose())
            cov_list.append(p.weight * R)
        cov_matrix = np.sum(cov_list, axis=0)

        return cov_matrix

    def normalize_step_weights(self):
        weights = self.get_weights()
        particles = self.particles
        total_weight = np.sum(weights)
        for p in particles:
            p.weight = p.weight / total_weight
        return None

    def compute_ess(self):
        weights = self.get_weights()
        if not np.isclose(np.sum(weights), 1):
            self.normalize_step_weights()
        return 1 / np.sum([w**2 for w in weights])

    def get_params(self, key):
        particles = self.particles
        return np.array([p.params[key] for p in particles])

    def get_param_dicts(self):
        particles = self.particles
        return [p.params for p in particles]

    def get_particles(self):
        return self.particles

    def resample(self):
        particles = self.particles
        num_particles = len(particles)
        weights = self.get_weights()
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
                    new_particles[-1].weight = uniform_weight
                    break
        self.particles = new_particles
        return None

    def print_particle_info(self, particle_num):
        particle = self.particles[particle_num]
        print '-----------------------------------------------------'
        print 'Particle: %s' % particle_num
        particle.print_particle_info()
        return None

    def plot_marginal(self, key, save=False, show=True,
                      prefix='marginal_'):
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
            ax.plot([p.params[key], p.params[key]], [0.0, p.weight])
            ax.plot(p.params[key], p.weight, 'o')
        if save:
            plt.savefig(prefix + key + '.png')
        if show:
            plt.show()
        plt.close(fig)
        return None

    def plot_pairwise_weights(self, param_names=None, labels=None,
                              save=False, show=True, param_lims=None,
                              label_size=None, tick_size=None, nbins=None,
                              prefix='pairwise'):
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

        weights = self.get_weights()
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
            sc = ax[key1 + '+' + key2].scatter(pkey1, pkey2, c=weights, vmin=0.0,
                                               vmax=rnd_to_sig(max(weights)))
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
