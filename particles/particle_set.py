import numpy as np
import copy

class ParticleSet():
    '''
    Class defining data structure of an SMC particle set. The set stores all
    particle instances at all temperature steps (i.e., the entire particle set
    is M x T where M is the total number of particles and T is the number of
    steps in the temperature schedule. 
    '''

    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.step = []
        self.current_step = 0
        self.nsteps = 0


    def add_empty_step(self,):
        self.step.append([])
        self.nsteps += 1


    def add_particle(self, particle, step_number):
        '''
        Add a single particle to a given step.
        '''
        if self.nsteps == 0:
            self.add_empty_step()
        if len(self.step[step_number]) >= self.num_particles:
            msg = 'Cannot add particle; new step length would exceed number '+\
                  'of total particles allowed.'
            raise ValueError(msg)
        # check to see if new step should be added
        if step_number == len(self.step)+1:
            self.add_empty_step()
        elif step_number > len(self.step)+1:
            raise ValueError('cannot skip steps, next step available for '+\
                             'creation is %s' % (len(self.step)+1))
        self.step[step_number].append(particle)

    
    def add_step(self, particle_list):
        '''
        Add an entire step to the set, providing a list of particles.
        '''
        if len(particle_list) != self.num_particles:
            raise ValueError('len(particle_list) must equal self.num_particles')
        self.step.append(particle_list)
        self.nsteps += 1


    def get_likes(self, step=-1):
        '''
        Returns a list of all particle likelihoods for a given step.
        '''
        return [np.exp(p.log_like) for p in self.step[step]]


    def get_log_likes(self, step=-1):
        '''
        Returns a list of all particle log likelihoods for a given step.
        '''
        return [p.log_like for p in self.step[step]]


    def get_mean(self, step=-1):
        '''
        Computes mean for a given step.
        '''
        param_names = self.step[step][0].params.keys()
        mean = dict()
        for pn in param_names:
            mean[pn] = []
            for p in self.step[step]:
                mean[pn].append(p.weight*p.params[pn])
            mean[pn] = np.sum(mean[pn])
        return mean


    def get_params(self, key, step=-1):
        '''
        Returns an array of param <key> with length = number of particles.
        '''
        particles = self.get_particles(step)
        return np.array([p.params[key] for p in particles])

    
    def get_param_dicts(self, step=-1):
        '''
        Returns a list of particle parameter dictionaries.
        '''
        particles = self.get_particles(step)
        return [p.params for p in particles]


    def get_particles(self, step=-1):
        '''
        Returns a list of all particles for a given step.
        '''
        return self.step[step]


    def get_weights(self, step=-1):
        '''
        Returns a list of all weights for a given step.
        '''
        return [p.weight for p in self.step[step]]


    def calculate_step_covariance(self, step=-1):
        '''
        Estimates the covariance matrix for a given step in the set.

        :param int step: step identifier, default is most recent (i.e., step=-1)
        '''
        particle_list = self.get_particles(step)
        param_names = particle_list[0].params.keys()

        # get mean vector
        means = np.array([self.get_mean(step)[pn] for pn in param_names])

        # get covariance matrix
        cov_list = []
        for p in particle_list:
            param_vector = np.array([p.params[pn] for pn in param_names])
            diff = param_vector-means
            diff = diff.reshape(-1, 1)
            R = np.dot(diff, diff.transpose())
            cov_list.append(p.weight*R)
        cov_matrix = np.sum(cov_list, axis=0)
        
        return cov_matrix


    def compute_ESS(self, step=-1):
        '''
        Computes the effective sample size (ESS) of a given step in the particle
        set.
        '''
        weights = self.get_weights(step)
        # make sure weights are normalized
        if not 0.9999999999 < np.sum(weights) < 1.0000000001:
            self.normalize_step_weights(step)
        return 1/np.sum([w**2 for w in weights])


    def copy(self, step=-1):
        '''
        Returns a copy of particle set at step (most recent step by default).
        '''
        return copy.deepcopy(self.step[-1])

    
    def normalize_step_weights(self, step=-1):
        '''
        Normalizes weights for all particles in a given step.

        :param int step: step identifier, default is most recent (i.e., step=-1)
        '''
        # TODO: find better way to handle log weights and non-log weights
        weights = self.get_weights(step)
        particles = self.get_particles(step)

        # if non-log weights
        if np.sum(weights) > 0:
            total_weight = np.sum(weights)
            for p in particles:
                p.weight = p.weight/total_weight

        # if log weights
        else:
            total_weight = np.sum(np.exp(weights))
            for p in particles:
                # normalized weights between 0 and 1 (unlog)
                p.weight = np.exp(p.weight-np.log(total_weight))


    def overwrite_step(self, step, particle_list):
        '''
        Overwrite an entire step of the set with the provided list of
        particles.
        '''
        print particle_list
        if len(particle_list) != self.num_particles:
            raise ValueError('len(particle_list) must equal self.num_particles')
        self.step[step] = particle_list


    def plot_marginal(self, key, step=-1, save=False, show=True,
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
        for p in self.step[step]:
            ax.plot([p.params[key], p.params[key]], [0.0, p.weight])
            ax.plot(p.params[key], p.weight, 'o')
        if save == True:
            plt.savefig(prefix+key+'.png')
        if show == True:
            plt.show()
        plt.close(fig)


    def plot_all_marginals(self, step=-1, save=False, show=True,
                           prefix='marginal_'):
        '''
        Plots marginal approximation for all parameters in the set.
        '''
        try:
            plt
        except:
            import matplotlib.pyplot as plt
        param_names = self.step[0][0].params.keys()
        for i, pn in enumerate(param_names):
            self.plot_marginal(key=pn, step=step, save=save, show=show,
                               prefix=prefix)


    def plot_pairwise_weights(self, step=-1, param_names=None, labels=None,
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
        particles = self.get_particles(step)

        # set up label dictionary
        if param_names == None:
            param_names = particles[0].params.keys()
        if labels == None:
            labels = param_names
        label_dict = {key: lab for key, lab in zip(param_names, labels)}
        if param_lims is not None:
            lim_dict = {key: l for key, l in zip(param_names, param_lims)}
        if nbins is not None:
            bin_dict = {key: n for key, n in zip(param_names, nbins)}
        L = len(param_names)

        # setup figure
        fig = plt.figure(figsize=[10*(L-1)/2,10*(L-1)/2])

        # create lower triangle to obtain param combos
        tril = np.tril(np.arange(L**2).reshape([L,L]),-1)
        ikeys = np.transpose(np.nonzero(tril)).tolist()
    
        # use lower triangle to id subplots
        tril = np.tril(np.arange((L-1)**2).reshape([L-1,L-1])+1)
        iplts = [i for i in tril.flatten() if i > 0]
    
        weights = self.get_weights(step)
        means = self.get_mean(step)
        for i in zip(iplts, ikeys):
            iplt = i[0]     # subplot index
            ikey1 = i[1][1] # key index for xparam
            ikey2 = i[1][0] # key index for yparam
            key1 = param_names[ikey1]
            key2 = param_names[ikey2]
            ax = {key1+'+'+key2:fig.add_subplot(L-1, L-1, iplt)}
            # get list of all particle params for key1, key2 combinations
            pkey1 = []
            pkey2 = []
            for p in particles:
                pkey1.append(p.params[key1])
                pkey2.append(p.params[key2])
            # plot parameter combos with weight as color
            rnd_to_sig = lambda x: round(x, -int(np.floor(np.log10(abs(x))))+1)
            sc = ax[key1+'+'+key2].scatter(pkey1, pkey2, c=weights, vmin=0.0,
                    vmax=rnd_to_sig(max(weights)))
            ax[key1+'+'+key2].axvline(means[key1], color='C1', linestyle='--')
            ax[key1+'+'+key2].axhline(means[key2], color='C1', linestyle='--')
            ax[key1+'+'+key2].set_xlabel(label_dict[key1])
            ax[key1+'+'+key2].set_ylabel(label_dict[key2])

            # if provided, set x y lims
            if param_lims is not None:
                ax[key1+'+'+key2].set_xlim(lim_dict[key1])
                ax[key1+'+'+key2].set_ylim(lim_dict[key2])
            # if provided set font sizes
            if tick_size is not None:
                ax[key1+'+'+key2].tick_params(labelsize=tick_size)
            if label_size is not None:
                ax[key1+'+'+key2].xaxis.label.set_size(label_size)
                ax[key1+'+'+key2].yaxis.label.set_size(label_size)
            # if provided, set x ticks
            if nbins is not None:
                ax[key1+'+'+key2].locator_params(axis='x', nbins=bin_dict[key1])
                ax[key1+'+'+key2].locator_params(axis='y', nbins=bin_dict[key2])

        fig.tight_layout()

        # colorbar
        if L <= 2:
            plt.colorbar(sc, ax=ax[key1+'+'+key2])
        else:
            ax1_position = fig.axes[0].get_position()
            ax3_position = fig.axes[2].get_position()
            y0 = ax1_position.y0
            x0 = ax3_position.x0
            w = 0.02
            h = abs(ax1_position.y1-ax1_position.y0)
            empty_ax = fig.add_axes([x0, y0, w, h])
            cb = plt.colorbar(sc, cax=empty_ax)
            if tick_size is not None:
                empty_ax.tick_params(labelsize=tick_size)

        if save==True:
            plt.savefig(prefix+'.png')
        if show==True:
            plt.show()
        plt.close(fig)


    def print_particle_info(self, step_num, particle_num):
        step = self.step[step_num]
        particle = step[particle_num]
        print '-----------------------------------------------------'
        print 'Step: %s' % step_num
        print 'Particle: %s' % particle_num
        particle.print_particle_info()


    def print_step_info(self, step_num=-1):
        step = self.step[step_num]
        print '-----------------------------------------------------'
        print 'Step: %s' % step_num
        for i, particle in enumerate(step):
            print '-----------------------------------------------------'
            print 'Particle: %s' % i
            particle.print_particle_info()
    

    def resample(self, step=-1, overwrite=True):
        '''
        Resamples a given step in the particle set based on normalized 
        weights. Assigns discrete probabilities to each particle (sum to 1),
        resample from this discrete distribution using the particle's copy()
        method.

        :param boolean overwrite: if True (default), overwrites current step
            with resampled step, else appends new step
        '''
        particles = self.get_particles(step)
        weights = self.get_weights(step)
        weights_cs = np.cumsum(weights)
        
        # intervals based on weights to use for discrete probability draw
        intervals = zip(np.insert(weights_cs, 0, 0)[:-1], weights_cs)

        # generate random numbers, iterate to find intervals for resample
        R = np.random.uniform(0, 1, [self.num_particles,])
        new_particles = []
        uniform_weight = 1./self.num_particles
        for r in R:
            for i, (a, b) in enumerate(intervals):

                if a <= r < b:
                    # resample
                    new_particles.append(particles[i].copy())
                    # assign uniform weight
                    new_particles[-1].weight = uniform_weight
                    break

        # overwrite or append new step
        if overwrite == True:
            self.overwrite_step(step, new_particles)
        else:
            self.add_step(new_particles)


    def save(self, filename='pset.p', mode='w'):
        #TODO: change to hdf5
        '''
        Saves particle set as a pickle file.
        '''
        import pickle
        with open(filename, mode) as pf:
            pickle.dump(self, pf)
