'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
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
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
 
'''
import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mpcolors
import numpy as np
import os
import pymc
from pymc.Matplot import plot as pymc_plot
from scipy.stats import gaussian_kde
import statsmodels.api as sm


'''
Defines plotting functions to be used in conjunction with the DamagePrognosis
module. Each function takes a set of arguments as outlined below in addition
to a filename ("fname") defining the image file name the plot is saved to.

    ======================================================================
    TYPE      | arguments
    ======================================================================
    Data      | x         : x data, array of shape (n x m) where n = # of
    <data>    |             observations and m = number of datasets
              | y         : y data, array of shape (n x m)
              | styles    : list of dash styles, must be length m
              | colors    : list of colors, must be length m
              | markers   : list of symbols, must be length m
              | xlabel    : label for x axis
              | ylabel    : label for y axis
    ----------------------------------------------------------------------
    Joint     | trace     : parameter trace object (i.e., chains)
    Pairwise  | keys      : keys for trace object (params to be plotted)
    <pairwise>| color     : color for scatter
              | xylim     : dict of axis limits for params of the form
              |             {'param':[lim0,lim1],...} (optional)
    ----------------------------------------------------------------------
    PDFs      | trace     : parameter trace object (i.e., chain)
    <pdf>     | keys      : keys for trace object (params to be plotted)
              | labels    : same as keys but should be in format for plot 
              |             labeling
              | color     : color of plot outlines (can be list)
              | facecolor : fill color for PDFs
              | alpha     : alpha for fill color (can be list)
              | plot_prior: True will plot the priors as dotted lines, 
              |             False will not plot any prior information
              | params    : required if plot_prior = True (dictionary of
              |             prior distributions used with prognosis)
              | xylim     : dict of axis limits for params of the form
              |             {'param':[lim0,lim1],...} (optional)
    ----------------------------------------------------------------------
    Residuals | model     : model instance with an evaluate method
    <residual>| params    : model parameters
              | data      : data for computation of residuals  
    ======================================================================
'''


def time_vs_observations(x, y, styles, colors, markers, xlabel, ylabel, fname):
    '''
    Simple plot of the data (e.g., time vs observations). Give data as column
    vectors (lists will be automatically converted). This allows for multiple
    columns representing multiple datasets.
    '''
    #TODO: refactor with list for x and y to allow for data of diff lengths
    print 'plotting data...'

    fig = plt.figure(figsize=[10, 10])
    # convert lists to column arrays
    if type(x) == list:
        x = np.array(x).reshape(-1,1)
    if type(y) == list:
        y = np.array(y).reshape(-1,1)

    # plot
    #fig = plt.figure(figsize=[10,10])
    shape = x.shape
    for i in xrange(shape[1]):
        plt.plot(x[:,i], y[:,i], linestyle=styles[i], color=colors[i], 
                 marker=markers[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)

    return plt.gcf()


def pdf(trace, keys, labels=None, color='0.2', facecolor='C0', line_alpha=1.0, 
        face_alpha=0.9, plot_prior=False, params=None, xylim=None, figsize=None,
        ylabel=None, nbins_x=3, nbins_y=6, fname='pdfs.png', truth=None):     
    '''
    Plots the probability distribution function of the parameters defined by
    "trace", "keys" and "labels" along with their associated chains obtained
    from MCMC sampling. 
    '''
    print 'plotting parameter chains/pdfs...'
    
    # set up labels if not provided
    if labels is None:
        labels = keys
    label_dict = {key: lab for key, lab in zip(keys, labels)}

    # handle extra keys that are not in trace
    i_keep = []
    for i, k in enumerate(keys):
        try:
            trace(k)
            i_keep.append(i)
        except KeyError:
            print 'param <%s> is not in trace; skipping this pdf plot.' % k
    keys = [keys[i] for i in i_keep]
    labels = [labels[i] for i in i_keep]

    # plot
    if figsize == None:
        fig = plt.figure(figsize=[10, 10*len(keys)/3])
    else:
        fig = plt.figure(figsize=figsize)

    ax_right = []; ax_left = []
    for i, key in enumerate(keys):
        
        #TODO: add check for length of lists:
        if type(facecolor) == list:
            facecolor = facecolor[i]
        if type(color) == list:
            color = color[i]
        if type(face_alpha) == list:
            face_alpha = face_alpha[i]
        if type(line_alpha) == list:
            line_alpha = line_alpha[i]
       
        # define left and right axes (left = chains, right = pdfs)
        ax_left += [fig.add_subplot(len(keys), 2, i*2+1)]
        ax_right += [fig.add_subplot(len(keys), 2, i*2+2)]
        
        # plot left
        ax_left[i].plot(trace(key)[:], color=color, 
                        alpha=line_alpha, linewidth=1)
        ax_left[i].set_ylabel(labels[i])
        ax_left[i].set_xlabel('Chain iteration')
        ax_left[i].locator_params(nbins=nbins_x, axis='x')
        ax_left[i].locator_params(nbins=nbins_y, axis='y')
        
        # plot right
        x = np.linspace(min(trace(key)[:]),
                        max(trace(key)[:]), 1000)
        y = gaussian_kde(trace(key)[:]).pdf(x)
        ax_right[i].fill_between(x, np.tile(0,y.shape), y, 
                                 facecolor=facecolor, alpha=face_alpha)
        ax_right[i].plot(x, y, color)
        ax_right[i].set_xlabel(labels[i])
        if ylabel == None:
            ax_right[i].set_ylabel('Probability density')
        else:
            ax_right[i].set_ylabel(ylabel)
        ax_right[i].locator_params(nbins=nbins_x, axis='x')
        ax_right[i].locator_params(nbins=nbins_y, axis='y')

        # plot prior as dotted line if requested
        if plot_prior == True:
            print 'plot priror = True'
            print params
            if params != None:                  
                print 'params != None = True'
                if params[key][0] == 'TruncatedNormal':
                    print 'truncatednorm = True'
                    predictive = pymc.TruncatedNormal('predictive', 
                                                      params[key][1],
                                                      params[key][2], 
                                                      params[key][3],
                                                      params[key][4])
                    model = pymc.Model({"pred":predictive})
                    mcmc = pymc.MCMC(model)
                    mcmc.sample(10000, 1000)
                    samples = mcmc.trace('predictive')[:]
                    print samples
    
                    kde = sm.nonparametric.KDEUnivariate(samples)
                    kde.fit()
                    x_prior = kde.support
                    y_prior = kde.density
                    ax_right[i].plot(x_prior, y_prior, '--', color='k')#color)


        if truth != None:
            if type(truth) == dict:
                ax_right[i].plot(truth[key], 0., 'k^')
            else:
                raise TypeError('truth must be dictionary w/ params as keys')

        # set parameter axis limits if provided
        if xylim != None:
            if key in xylim:
                ax_right[i].set_xlim(xylim[key])
                ax_left[i].set_ylim(xylim[key])
        else:
            ax_right[i].set_ylim(ymin=0)
            ax_left[i].set_xlim([0, len(trace(key)[:])])

    fig.tight_layout() 
    plt.savefig(fname, dpi=300)

    return plt.gcf()


def pairwise(trace, keys, labels=None, color='C0', xylim=None,
             fname='pairwise.png', fig='new', nbins=None, label_font_size=None,
             tick_label_font_size = None):
    '''
    Pairwise plots of all sampled parameters defined by "trace", "keys", and
    "labels." 
    '''
    print 'plotting pairwise samples...'

    #TODO: fix this (bad way to handle input of lists)
    # if trace is a list, assume a list of trace objects
    if type(trace) is not list:
        traces = [trace]
        colors = [color]
    else:
        traces = trace
        colors = color

    # set up labels if not provided
    if labels is None:
        labels = keys
    label_dict = {key: lab for key, lab in zip(keys, labels)}

    # handle extra keys that are not in trace
    i_keep = []
    for i, k in enumerate(keys):
        try:
            print k
            print traces[0]
            traces[0](k)
            i_keep.append(i)
        except KeyError:
            print 'param <%s> is not in trace; skipping this pairwise plot.' % k
    keys = [keys[i] for i in i_keep]
    labels = [labels[i] for i in i_keep]
    
    # ensure that number of params > 1
    if len(keys) <= 1:
        print 'number of parameters to plot <= 1; cannot plot pairwise.'
        return 0

    # plot
    L = len(keys)
    
    if fig == 'new':
        fig = plt.figure(figsize=[10*(L-1)/2,10*(L-1)/2])
    
    # create lower triangle to obtain param combos
    tril = np.tril(np.arange(L**2).reshape([L,L]),-1)
    ikeys = np.transpose(np.nonzero(tril)).tolist()
    
    # use lower triangle to id subplots
    tril = np.tril(np.arange((L-1)**2).reshape([L-1,L-1])+1)
    iplts = [i for i in tril.flatten() if i > 0]
    
    for j, trace in enumerate(traces):
        for i in zip(iplts, ikeys):
        
            iplt = i[0]     # subplot index
            ikey1 = i[1][1] # key index for xparam
            ikey2 = i[1][0] # key index for yparam
        
            key1 = keys[ikey1]
            key2 = keys[ikey2]
        
            ax = {key1+'+'+key2:fig.add_subplot(L-1, L-1, iplt)}
            ax[key1+'+'+key2].plot(trace(key1)[:], trace(key2)[:], 'o',
                                   color=colors[j], alpha=0.5)
            # plot mean as vertical and horizontal lines
            print 'DEBUG: %s median = %s' % (key1, np.median(trace(key1)[:]))
            print 'DEBUG: %s mean = %s' % (key1, np.mean(trace(key1)[:]))
            print 'DEBUG: %s var = %s' % (key1, np.var(trace(key1)[:]))
            ax[key1+'+'+key2].axvline(np.mean(trace(key1)[:]), color='C1',
                                      linestyle='--')
            ax[key1+'+'+key2].axhline(np.mean(trace(key2)[:]), color='C1',
                                      linestyle='--')
            # labels and ticks
            ax[key1+'+'+key2].set_xlabel(label_dict[key1])
            ax[key1+'+'+key2].set_ylabel(label_dict[key2])

            # if provided, set axes limits with params['xylim']                
            if xylim != None:
                if key1 in xylim.keys():
                    ax[key1+'+'+key2].set_xlim(xylim[key1])
                if key2 in xylim.keys():
                    ax[key1+'+'+key2].set_ylim(xylim[key2])

            # if provided, set font size of ticks
            if tick_label_font_size is not None:
                ax[key1+'+'+key2].tick_params(labelsize=tick_label_font_size)
            # if provided, set font size of labels
            if label_font_size is not None:
                ax[key1+'+'+key2].xaxis.label.set_size(label_font_size)
                ax[key1+'+'+key2].yaxis.label.set_size(label_font_size)
            # if provided, set nbins per axis
            if nbins is not None:
                ax[key1+'+'+key2].locator_params(axis='x', nbins=nbins[key1])
                ax[key1+'+'+key2].locator_params(axis='y', nbins=nbins[key2])
    
    print 'DEBUG: cov = %s' % np.cov(np.vstack([trace(key)[:] for key in keys]))

    fig.tight_layout()
    plt.savefig(fname)

    return plt.gcf()


def residuals(model, params, data, fname):
    '''
    Plots the residuals (comparing the evaluation using "model" given "params"
    to "data." Residuals = data-evaluation
    '''
    print 'plotting residuals...'
    
    fig = plt.figure(figsize=[10, 10])
    f = model.evaluate(params)
    residuals = np.array(data).flatten()-np.array(f).flatten()
    plt.plot(residuals, 'xb')
    plt.xlabel('model outputs')
    plt.ylabel('residuals')

    fig.tight_layout()
    plt.savefig(fname)

    return plt.gcf()


def pdf_series(sample_array, color='g', figsize=[10, 10], fade=True,
               fname='pdf_series', xmin=None, xmax=None, ymin=None, ymax=None,
               xlabel='Parameter', ylabel='Probability density',
               linewidth=2.0, fade_linewidth=1.0, numpts_support=1000,
               nbins_xaxis=5, truth=None, xsci=False, ysci=False):
    '''
    Gaussian KDE used to fit smooth pdf to the rows of "sample_array."
    The number of plots generated = number of rows in sample_array. Each
    plot will add the next pdf (retaining all previous). If "fade" == True, 
    then the previous pdfs will be grayed out, with increasing transparency
    based on how many iterations have passed. Input an array of truth values
    with length = number of rows in sample_array to plot a "truth" marker.
    '''
    # TODO: add more checks/tests, add doc to top
    # get shape of arrray    
    r, c = sample_array.shape

    # x limits
    if xmin == None:
        xmin = min(sample_array.flatten())
    if xmax == None:
        xmax = max(sample_array.flatten())

    # iterate through sample_array rows
    for i, samples in enumerate(sample_array):
        # kde fit
        support = np.linspace(xmin, xmax, numpts_support)
        density = gaussian_kde(samples).pdf(support)

        if i == 0:
            # initialize x and y arrays
            x = support
            y = density
        else:
            # append to x and y arrays
            x = np.vstack((x, support))
            y = np.vstack((y, density))
        
        # initialize fade
        fade_alpha  = np.linspace(0.6/(x.shape[0]-1), 0.6, x.shape[0]-1)

        # plot current pdf iteration 
        fig = plt.figure(figsize=figsize)
        plt.plot(support, density, color=color, linewidth=linewidth)

                # apply fade if requested, else plot with defaults
        if fade == True:
            for j in xrange(i):
                plt.plot(x[j], y[j], alpha=fade_alpha[j], color='0.2',
                         linewidth=fade_linewidth)
        else:
            for j in xrange(i):
                plt.plot(x[j], y[j], linewidth=linewidth)

        if truth != None:
            plt.plot(truth[i], 0, '^', color='k', clip_on=False)

        # x limits
        ax = plt.gca()
        ax.set_xlim([xmin, xmax])

        # y limits
        if ymin == None:
            ymin = plt.gca().get_ylim()[0]
        if ymax == None:
            ymax = plt.gca().get_ylim()[1]
        ax.set_ylim([ymin, ymax])

        # scientific notation
        if ysci == True:
            # set y scientific if requested
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 

        if xsci == True:
            # set y scientific if requested
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 

        # plot formatting
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.locator_params(nbins=nbins_xaxis)
        plt.tight_layout()
        plt.savefig(fname+str(i), dpi=300)
        plt.close('all')

    return plt.gcf()


def plot_parameter_updates(mcmc_filename, dir_list, params_to_plot,
                           param_labels, save_dir, save_filename, dashes=None,
                           colors = None, figsize=[10,10], xlog_params=None,
                           ylog_params=None, dpi=100):
    '''
    Plots parameter updates for parameters stored in a PyMC database object with
    keys given in params_to_plot. Updates should be in separate directories,
    which are given by dir_list. Parameters which should be plotted on a log
    scale are listed in xlog_params. Parameters for which probability should
    be on a log scale are listed in ylog_params. Both are optional. 
    '''
    print 'plotting parameter updates...'

    # get number of params
    n = len(params_to_plot)

    # set up default dashes if not given
    if dashes is None:
        dashes = np.tile([1000, 1], [len(dir_list), 1]).tolist()

    # set up default colors if not given
    if colors is None:
        from matplotlib import colors as mcolors
        colors = mcolors.cnames.keys()

    # set up fig
    fig = plt.figure(figsize=figsize)
    rows = np.ceil(np.sqrt(n)) 
    columns = np.ceil((n)/rows)
    ax  = [fig.add_subplot(rows, columns, i+1) for i in xrange(n)]
    
    # iterate through directories, store chains, get max/min x
    db = []
    xmin = dict()
    xmax = dict()
    for i, d in enumerate(sorted(dir_list)):
        # load mcmc
        db.append(pymc.database.pickle.load(os.path.realpath(d)+'/'+\
                  mcmc_filename))
        for j, p in enumerate(params_to_plot):
            if i == 0:
                xmin[p] = min(db[i].trace(p)[:])
                xmax[p] = max(db[i].trace(p)[:])
            else:
                xmin[p] = min(xmin[p], min(db[i].trace(p)[:]))
                xmax[p] = max(xmax[p], max(db[i].trace(p)[:]))

    # plot
    handles = []
    for i, d in enumerate(db):
        # plot each trace for each chain
        for j, p in enumerate(params_to_plot):
            if abs(xmax[p]-xmin[p]) > 1:
                x_decimal = 0
            else:
                x_decimal = int(np.ceil(abs(np.log10(xmax[p]-xmin[p]))))
            xmax_temp = np.round(xmax[p], x_decimal)
            xmin_temp = np.round(xmin[p], x_decimal)
    
            trace = d.trace(p)[:]
            x = np.linspace(xmin_temp, xmax_temp, 100)
            kde = gaussian_kde(trace)
            y = kde.pdf(x)

            if j == 0:
                handles += ax[j].plot(x, y, dashes=dashes[i], color=colors[i])
            else:
                ax[j].plot(x, y, dashes=dashes[i], color=colors[i])
    
            ax[j].set_xlabel(param_labels[j])
            ax[j].set_ylabel('Probability Density')
    
            ax[j].locator_params(axis='x', nbins=5)
    
    # tight layout
    plt.tight_layout()
    
    # set up legend
    labels  = ['Data Interval '+str(i+1) for i, d in enumerate(dir_list)]
    legend  = ax[0].legend(handles, labels, handlelength=2.0, loc=2, ncol=1)

    # log any axis as requested
    if any([xlog_params, ylog_params]):
        for i, param in enumerate(params_to_plot):
            if param in xlog_params:
                ax[i].set_xscale('symlog')
            if param in ylog_params:
                ax[i].set_yscale('symlog')
    
    # save figure
    plt.savefig(os.path.realpath(save_dir)+'/'+save_filename, dpi=dpi)

    return plt.gcf()


def plot_pymc_autocorrelation(mcmc_database, keys, working_dir):
    for key in keys:
        trace = mcmc_database.trace(key)
        pymc_plot(trace, 'png', path=working_dir)
    return None


def plot_geweke_scores(mcmc_database, keys, working_dir):
    for key in keys:
        print('plotting geweke scores for %s...' % key)
        scores = pymc.geweke(mcmc_database.trace(key)[:])
        pymc.Matplot.geweke_plot(scores, name=key, path=working_dir)
    return None


def set_rcParams(params):
    '''
    Takes a dictionary of rcparam keys : values. Updates system rcparams
    according to the entries in the dictionary.
    '''
    rcParams.update(params)
