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

import imp
import numpy as np
import matplotlib.pyplot as plt


def _mpi_decorator(func):
    def wrapper(self, *args, **kwargs):
        """
        Detects whether multiple processors are available and sets
        self.number_CPUs and self.cpu_rank accordingly. Only calls decorated
        function using rank 0.
        """
        try:
            imp.find_module('mpi4py')

            from mpi4py import MPI
            comm = MPI.COMM_WORLD.Clone()

            size = comm.size
            rank = comm.rank
            comm = comm

        except ImportError:

            size = 1
            rank = 0
            comm = SingleRankComm()

        if rank == 0:
            func(self, *args, **kwargs)

    return wrapper


#@_mpi_decorator
#def plot_marginal(self, key, save=False, show=True,
#                  prefix='marginal_'):  # pragma no cover
#    '''
#    Plots a single marginal approximation for param given by <key>.
#    '''
#    try:
#        plt
#    except:
#        import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for p in self.particles:
#        ax.plot([p.params[key], p.params[key]], [0.0, np.exp(p.log_weight)])
#        ax.plot(p.params[key], np.exp(p.log_weight), 'o')
#    if save:
#        plt.savefig(prefix + key + '.png')
#    if show:
#        plt.show()
#    plt.close(fig)
#    return None

def plot_pairwise(samples, weights=None, param_names=None,
                  param_labels=None, save=False, show=True,
                  param_limits=None, label_size=None, tick_size=None,
                  num_xbins=None, prefix='pairwise'):  # pragma no cover
    '''
    Plots pairwise distributions of all parameter combos. Color codes each
    by weight if provided.
    '''

    # set up label dictionary
    L = samples.shape[1]

    if param_names is None:
        param_names = ['param{}'.format(i) for i in range(L)]

    if param_labels is None:
        param_labels = param_names
    label_dict = {key: lab for key, lab in zip(param_names, param_labels)}

    if param_limits is not None:
        lim_dict = {key: l for key, l in zip(param_names, param_limits)}

    if num_xbins is not None:
        bin_dict = {key: n for key, n in zip(param_names, num_xbins)}

    # setup figure
    fig = plt.figure(figsize=[10 * (L - 1) / 2, 10 * (L - 1) / 2])

    # create lower triangle to obtain param combos
    tril = np.tril(np.arange(L**2).reshape([L, L]), -1)
    ikeys = np.transpose(np.nonzero(tril)).tolist()

    # use lower triangle to id subplots
    tril = np.tril(np.arange((L - 1)**2).reshape([L - 1, L - 1]) + 1)
    iplts = [i for i in tril.flatten() if i > 0]

    # compute means
    if weights is None:
        means = np.mean(samples, axis=0)
    else:
        means = np.sum(samples * weights, axis=0)

    # plot
    for i in zip(iplts, ikeys):

        iplt = i[0]     # subplot index
        ikey1 = i[1][1]  # key index for xparam
        ikey2 = i[1][0]  # key index for yparam
        key1 = param_names[ikey1]
        key2 = param_names[ikey2]
        ax = {key1 + '+' + key2: fig.add_subplot(L - 1, L - 1, iplt)}

        # get list of all particle params for key1, key2 combinations
        pkey1 = samples[:, ikey1]
        pkey2 = samples[:, ikey2]

        # plot parameter combos with weight as color
        def rnd_to_sig(x):
            return np.round(x, -int(np.floor(np.log10(abs(x)))) + 1)

        if weights is None:
            alpha = 0.5
            colors = 'C0'
            vmax = None
        else:
            alpha = None
            colors = weights.flatten()
            vmax = rnd_to_sig(max(weights))

        sc = ax[key1 + '+' + key2].scatter(pkey1, pkey2, c=colors, vmin=0.0,
                                           vmax=vmax, alpha=alpha)

        ax[key1 + '+' + key2].axvline(means[ikey1], color='C1',
                                      linestyle='--')
        ax[key1 + '+' + key2].axhline(means[ikey2], color='C1',
                                      linestyle='--')

        ax[key1 + '+' + key2].set_xlabel(label_dict[key1])
        ax[key1 + '+' + key2].set_ylabel(label_dict[key2])

        # if provided, set x y lims
        if param_limits is not None:
            ax[key1 + '+' + key2].set_xlim(lim_dict[key1])
            ax[key1 + '+' + key2].set_ylim(lim_dict[key2])
        else:
            deltax = abs(pkey1.max() - pkey1.min())
            deltay = abs(pkey2.max() - pkey2.min())
            ax[key1 + '+' + key2].set_xlim(pkey1.min() - 0.05 * deltax,
                                           pkey1.max() + 0.05 * deltax)
            ax[key1 + '+' + key2].set_ylim(pkey2.min() - 0.05 * deltay,
                                           pkey2.max() + 0.05 * deltay)

        # if provided set font sizes
        if tick_size is not None:
            ax[key1 + '+' + key2].tick_params(labelsize=tick_size)
        if label_size is not None:
            ax[key1 + '+' + key2].xaxis.label.set_size(label_size)
            ax[key1 + '+' + key2].yaxis.label.set_size(label_size)

        # if provided, set x ticks
        if num_xbins is not None:
            ax[key1 + '+' + key2].locator_params(axis='x',
                                                 num_xbins=bin_dict[key1])
            ax[key1 + '+' + key2].locator_params(axis='y',
                                                 num_xbins=bin_dict[key2])

    fig.tight_layout()

    # colorbar
    if weights is not None:
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
