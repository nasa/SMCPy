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
from copy import copy
import functools
import numpy as np
import os
import pickle
import pymc

from step_methods import DelayedRejectionAdaptiveMetropolis
from step_methods import SMC_Metropolis
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from mcmc_plots import time_vs_observations, pdf, pairwise, residuals


class MCMCSampler:
    '''
    Class for MCMC sampling; based on PyMC (https://github.com/pymc-devs/pymc).
    Uses Bayesian inference, MCMC, and a model to estimate parameters with
    quantified uncertainty based on a set of observations. 
    '''

    def __init__(self, data, model, params, working_dir='./',
                 storage_backend='pickle'):
        '''
        Set data and model class member variables, set working directory,
        and choose storage backend.

        :param data: data to use to inform MCMC parameter estimation;
            should be same type/shape/size as output of model.
        :type data: array_like
        :param model: model for which parameters are being estimated via MCMC;
            should return output in same type/shape/size as data. A baseclass
            exists in the Model module that is recommended to define the model
            object; i.e., model.__bases__ == <class model.Model.Model>,)
        :type model: object
        :param params: map where keys are the unknown parameter 
            names (string) and values are lists that define the prior
            distribution of the parameter [dist name, dist. arg #1, dist. arg
            #2, etc.]. The distribution arguments are defined in the PyMC
            documentation: https://pymc-devs.github.io/pymc/.
        :type params: dict
        :storage_backend: determines which format to store mcmc data,
            see self.avail_backends for a list of options.
        :type storage_backend: str
        '''

        self.params = params
        self.model = model
        self.working_dir = working_dir

        # Check for valid storage backend
        self.avail_backends = ['pickle', 'ram', 'no_trace', 'txt', 'sqlite',
                               'hdf5']
        if storage_backend not in self.avail_backends:
            msg = 'invalid backend option for storage: %s.' % storage_backend
            raise NameError(msg)
        else:
            self.storage_backend = storage_backend
            if storage_backend == 'pickle':
                self.loader = pymc.database.pickle 
                self.backend_ext = '.p'
            elif storage_backend == 'ram':
                self.loader = pymc.database.ram 
            elif storage_backend == 'no_trace':
                self.loader = pymc.database.no_trace 
            elif storage_backend == 'txt':
                self.loader = pymc.database.txt
                self.backend_ext = '.txt'
            elif storage_backend == 'sqlite':
                self.loader = pymc.database.sqlite
                self.backend_ext = '.db'
            elif storage_backend == 'hdf5':
                self.loader = pymc.database.hdf5
                self.backend_ext = '.h5'

        # verify working dir exists
        self._verify_working_dir()

        # Number of data points and parameters, respectively
        self.n = len(data)
        self.p = len(params)

        # Verify data format - for now, must be a list of crack lengths
        self.data = self._verify_data_format(data) # TODO: allow for samples

        # Initialize pymc model and MCMC database
        self.pymc_mod = None
        self.db = None

        self._initialize_plotting()

        
    def fit(self, q0=None, plot_residuals=False, plot_fit=False,
            opt_method='L-BFGS-B', repeats=1, save_results=False,
            fname='opt_results.p'):
        '''
        Fits the deterministic model given in self.model to the data in
        self.data using ordinary least squares regression. Returns a parameter
        map containing the optimized model parameters and the associated sum
        of squared error.

        The optional input, q0, should be an initial guess for the parameters
        in the form of a parameter map (dict).

        The optional input repeats dictates the number of times the optimization
        process is repeated, each time using the previous result as the new q0.
        '''
        # check if KDE distribution is in parameters
        if any(['KDE' in value for value in self.params.values()]):
            raise ValueError('KDE type distributions cannot be used with fit()')
        # ensure repeats >= 1
        if repeats < 1:
            raise ValueError('repeats must be >= 1.')

        # labels are alphabetically sorted parameter keys
        labels = sorted(self.params.keys(), key=str.lower)

        # limits are defined by the prior distribution in self.
        limits = self._get_limits()

        # bounds are 0 to 1 for all normalized parameters
        bounds = tuple((0.0, 1.0) for i in labels)

        # Initial guess
        if q0 == None:
            # if none, iterate over labels, and use midpoint of bounds 
            qnorm = [0.5 for key in labels]

        else:
            # Normalize initial guess:
            qnorm = self._true2norm(q0, labels, limits)

        for i in xrange(repeats):
            # Ordinary least squares regression
            res = minimize(self._ols, qnorm, method=opt_method,
                           args=(labels, limits), bounds=bounds)
            qnorm = res.x
            ssq_opt = res.fun
            print res

        # set optimized result
        qnorm_opt = qnorm

        # unNormalize
        q_opt = self._norm2true(qnorm_opt, labels, limits)
        print 'q_opt = %s' % q_opt
        print 'ssq_opt = %s' % ssq_opt

        # plot residuals if requested
        if plot_residuals == True:
            self.plot_residuals(self.model, q_opt, 
                                self.data, self.working_dir+'/residuals.png')

        if plot_fit == True:
            x = np.tile(np.arange(len(self.data)), [2, 1]).transpose()
            y = self.model.evaluate(q_opt)
            y_len = len(y)
            d_len = len(self.data)
            self.plot_data(x, np.c_[y, self.data],
                           styles=['-']*y_len+['']*d_len,
                           colors=['k']*y_len+['r']*d_len,
                           markers=['']*y_len+['o']*d_len,
                           xlabel='data index', ylabel='data/fit variable',
                           fname=self.working_dir+'/fit.png')

        # save results if requested
        if save_results == True:
            with open(os.path.join(self.working_dir, fname), 'w') as orf:
                pickle.dump({'q_opt': q_opt, 'ssq_opt': ssq_opt}, orf)

        return q_opt, ssq_opt


    def pymcplot(self):
        '''
        Generates a pymc plot for each parameter in self.. This plot
        includes a trace, histogram, and autocorrelation plot. For more control
        over the plots, see MCMCplots module. This is meant as a diagnostic
        tool only.
        '''
        from pymc.Matplot import plot as pymc_plot
        for name in self.params.keys():
            pymc_plot(self.db.trace(name), format='png', path=self.working_dir)
        print 'pymc plots generated for  = %s' % self.params.keys()


    def sample(self, num_samples, burnin, step_method='adaptive', interval=1000,
               delay=0, tune_throughout=False, scales=None, cov=None, thin=1,
               phi=None, verbose=0):
        '''
        Initiates MCMC sampling of posterior distribution using the
        model defined using the generate_pymc_model method. Sampling is
        conducted using the PyMC module. Parameters are as follows:
        
            - num_samples : number of samples to draw (int)
            - burnin : number of samples for burn-in (int)
            - adaptive : toggles adaptive metropolis sampling (bool)
            - step_method : step method for sampling; options are:
                  o adaptive - regular adaptive metropolis
                  o DRAM - delayed rejection adaptive metropolis
                  o metropolis - standard metropolis aglorithm
            - interval : defines frequency of covariance updates (only
                  applicable to adaptive methods)
            - delay : how long before first cov update occurs (only applicable
                  to adaptive methods)
            - tune_throughout : True > tune proposal covariance even after
                  burnin, else only tune proposal covariance during burn in
            - scales : scale factors for the diagonal of the multivariate
                  normal proposal distribution; must be dictionary with
                  keys = .keys() and values = scale for that param.
            - phi : cooling step; only used for SMC sampler
        '''
        # Check if pymc_mod is defined
        if self.pymc_mod == None:
            raise NameError('Cannot sample; self.pymc_mod not defined!')

        # Setup pymc sampler (set to output results as pickle file
        if self.storage_backend != 'ram':
            dbname = self.working_dir+'/mcmc'+self.backend_ext
            if self.storage_backend == 'hdf5':
                self._remove_hdf5_if_exists(dbname)
        else:
            dbname = None
        self.MCMC = pymc.MCMC(self.pymc_mod, db=self.storage_backend, 
                              dbname=dbname) #TODO: dbmode option

        # Set  as random variables (stochastics)
        # TODO rewrite using .index(param name)
        parameter_RVs = [self.pymc_mod[i] for i in xrange(self.p)]

        # Set scales
        if scales != None:
            if len(scales) != len(parameter_RVs):
                err_msg = 'len(scales) must be equal to num '
                raise ValueError(err_msg)
            scales = {rv: scales[rv.__name__] for rv in parameter_RVs}
        else:
            scales = None

        # assign step method
        if step_method.lower() == 'adaptive':
            print 'adaptation delay = %s' % delay
            print 'adaptation interval = %s' % interval
            self.MCMC.use_step_method(pymc.AdaptiveMetropolis, parameter_RVs,
                              shrink_if_necessary=True, interval=interval,
                              delay=delay, scales=scales, cov=cov,
                              verbose=verbose)

        elif step_method.lower() == 'dram':
            print 'adaptation delay = %s' % delay
            print 'adaptation interval = %s' % interval
            self.MCMC.use_step_method(DelayedRejectionAdaptiveMetropolis,
                              parameter_RVs, shrink_if_necessary=True,
                              interval=interval, delay=delay, scales=scales,
                              cov=cov, verbose=verbose) 

        elif step_method.lower() == 'smc_metropolis':
            self.MCMC.use_step_method(SMC_Metropolis, parameter_RVs, cov=cov,
                                      phi=phi, verbose=verbose) 

        elif step_method.lower() == 'metropolis':
            # if no scales provided,
            if scales is None:
                scales = dict()
                for rv in parameter_RVs:
                    scales[rv] = abs(rv.value/10.)
            # set Metropolis step method for each random variable
            for RV, scale in scales.iteritems():
                self.MCMC.use_step_method(pymc.Metropolis, RV, scale=scale,
                                          verbose=verbose)

        else:
            raise KeyError('Unknown step method passed to sample()')

        # sample
        if tune_throughout:
            print 'tuning proposal distribution throughout mcmc'
        self.MCMC.sample(num_samples, burnin, tune_throughout=tune_throughout,
                         thin=thin, verbose=verbose)
        #self.MCMC.db.close() TODO
        self._enable_trace_plotting()
        self.MCMC.db.commit()


    @staticmethod
    def _remove_hdf5_if_exists(dbname):
        if os.path.exists(dbname):
            print 'database %s exists; overwriting...' % dbname
            os.remove(dbname)
        return None


    def generate_pymc_model(self, q0=None, ssq0=None, std_dev0=None,
                            fix_var=False, model_output_stored=False):
        '''
        PyMC stochastic model generator that uses the parameter dictionary,
        self. and optional inputs:
            
            - q0        : a dictionary of initial values corresponding to keys 
                          in 
            - std_dev0  : an estimate of the initial standard deviation
            - ssq0      : the sum of squares error using q0 and self.data. Only
                          used if initial var, var0, is None.
            - fixed_var : determines whether or not variance will be sampled
                          (i.e., fixed_var == False) or fixed. 
        '''
        # Set up pymc objects from self.
        parents, pymc_mod, pymc_mod_order = self.generate_pymc_(self, q0)

        # Assign std_dev as random variable or fix
        if fix_var == True and std_dev0 != None:
            precision = 1./std_dev0**2
            pymc_mod_addon = []
            pymc_mod_order_addon = []
        elif fix_var == True and std_dev0 == None and ssq0 == None:
            raise ValueError('If fix_var == True, standard deviation, std_dev0',
                             ' or ssq0 must be specified!')
        elif fix_var == True and ssq0 != None and std_dev0 == None:
            std_dev0 = np.sqrt(ssq0/(self.n-self.p))
            print 'Variance fixed at an estimated value of %s' %std_dev0**2
            precision = 1./std_dev0**2
            pymc_mod_addon = []
            pymc_mod_order_addon = []
        else:
            print 'Standard deviation will be sampled via MCMC'
            std_dev = pymc.Uniform('std_dev', lower=0, upper=1000)

            # Since variance will be sampled, need to set initial value
            # if ssq0 provided, calculated std_dev0
            if std_dev0 == None and ssq0 != None:
                if self.n <= self.p:
                    raise TypeError('Length of data <= number of parameters! ',
                                    'Variance approximation is invalid. More ',
                                    'data is required, or an initial std dev ',
                                    'should be given (can be None).')
                std_dev0 = np.sqrt(ssq0/(self.n-self.p))
            # if ssq0 and std_dev0 not provided, set default value 
            elif std_dev0 == None and ssq0 == None:
                std_dev0 = 0.1
            else:
                print 'Setting initial estimate of std_dev = %s' %std_dev0
            std_dev.value = std_dev0

            # assign variance as random variable or fixed value
            @pymc.deterministic(plot=False)
            def var(std_dev=std_dev):
                return std_dev**2

            # Set up precision random variable
            @pymc.deterministic(plot=False)
            def precision(var=var):
                return 1.0/var
        
            # add to pymc_mod
            pymc_mod_addon = [precision, std_dev, var] # var must be last
            pymc_mod_order_addon = ['precision', 'std_dev', 'var']

        # Define deterministic model
        model = pymc.Deterministic(self.model.evaluate, name='model_RV', 
                                   doc='model_RV', parents=parents, 
                                   trace=model_output_stored, plot=False)

        # Posterior (random variable, normally distributed, centered at model)
        results = pymc.Normal('results', mu=model, tau=precision,
                              value=self.data, observed=True)

        # Assemble model and return
        pymc_mod += [model, results] + pymc_mod_addon # var is last
        pymc_mod_order += ['model', 'results'] + pymc_mod_order_addon 
        self.pymc_mod = pymc_mod
        self.pymc_mod_order = pymc_mod_order


    def generate_pymc_(self, params, q0=None):
        '''
        Creates PyMC objects for each param in  dictionary

        NOTE: the second argument for normal distributions is VARIANCE

        Prior option:
            An arbitrary prior distribution derived from a set of samples (e.g.,
            a previous mcmc run) can be passed with the following syntax:

                 = {<name> : ['KDE', <pymc_database>, <param_names>]}

            where <name> is the name of the distribution (e.g., 'prior' or
            'joint_dist'), <pymc_database> is the pymc database containing the
            samples from which the prior distribution will be estimated, and
            <param_names> are the children parameter names corresponding to the
            dimension of the desired sample array. This method will use all
            samples of the Markov chain contained in <pymc_database> for all
            traces named in <param_names>. Gaussian kernel-density estimation
            is used to derive the joint parameter distribution, which is then
            treated as a prior in subsequent mcmc analyses using the current
            class instance. The parameters named in <param_names> will be
            traced as will the multivariate distribution named <name>.
        '''
        pymc_mod = []
        pymc_mod_order = []
        parents = dict()

        # Iterate through , assign prior distributions
        for key, args in self.params.iteritems():
            # Distribution name should be first entry in [key]
            dist = args[0]

            if dist == 'Normal':
                if q0 == None:
                    RV = [pymc.Normal(key, mu=args[1], tau=1/args[2])]
                else:
                    RV = [pymc.Normal(key, mu=args[1], tau=1/args[2],
                          value=q0[key])]
            elif dist == 'Uniform':
                if q0 == None:
                    RV = [pymc.Uniform(key, lower=args[1], upper=args[2])]
                else:
                    RV = [pymc.Uniform(key, lower=args[1], upper=args[2], 
                          value=q0[key])]
            elif dist == 'DiscreteUniform':
                if q0 == None:
                    RV = [pymc.DiscreteUniform(key, lower=args[1],
                                               upper=args[2])]
                else:
                    RV = [pymc.DiscreteUniform(key, lower=args[1],
                                               upper=args[2], value=q0[key])]
            elif dist == 'TruncatedNormal':
                if q0 == None:
                    RV = [pymc.TruncatedNormal(key, mu=args[1], tau=1/args[2],
                          a=args[3], b=args[4])]
                else:
                    RV = [pymc.TruncatedNormal(key, mu=args[1], tau=1/args[2],
                          a=args[3], b=args[4], value=q0[key])]
            elif dist == 'KDE':
                kde = multivariate_kde_from_samples(args[1], args[2])
                kde_rv, rvs = self._create_kde_stochastic(kde, key, args[2])
                if q0 != None:
                    kde_rv.value = q0
                RV = [kde_rv]
                for rv_key, rv_value in rvs.iteritems():
                    parents[rv_key] = rv_value
                    RV.append(rv_value)
            else:
                raise KeyError('The distribution "'+dist+'" is not supported.')

            parents[key] = RV[0]
            pymc_mod_order.append(key)
            pymc_mod += RV
        
        return parents, pymc_mod, pymc_mod_order


    def save_model(self, fname='model.p'):
        '''
        Saves model in pickle file with name working_dir + fname.
        '''
        # store model
        model = {'model':self.model}

        # dump
        with open(self.working_dir+'/'+fname, 'w') as f:
            pickle.dump(model, f)


    def _verify_working_dir(self):
        '''
        Ensure specified working directory exists.
        '''
        if not os.path.isdir(self.working_dir):
            print 'Working directory does not exist; creating...'
            os.mkdir(self.working_dir)
            self.working_dir = os.path.realpath(self.working_dir)


    def _verify_data_format(self, data):
        '''
        Ensures that data is a single list.
        '''
        # For now, data should be a list of floats (e.g., of crack lengths)
        if type(data) == list or type(data) == np.ndarray:
            return np.array(data)
        else:
            raise TypeError('Data must be a single list of floats.')


    def _create_kde_stochastic(self, kde, kde_name, param_names):
        '''
        Creates custom pymc stochastic object based on a multivariate kernel
        density estimate (kde should be kde object from scipy).
        '''
        # build kde stochastic
        logp = lambda value: kde.logpdf(value)
        random = lambda value: kde.resample(1).flatten()
        KDE = pymc.Stochastic(logp = logp,
                              doc = 'multivariate KDE',
                              value=random(0),
                              name=kde_name,
                              parents=dict(),
                              random=random,
                              trace=True,
                              dtype=float,
                              observed=False,
                              plot=True)

        # build deterministics dependent on kde stochastic
        rvs = dict() 
        eval_func_dict = dict()
        for i, pn in enumerate(param_names):
            eval_func_dict[pn] = lambda i=i, **kwargs: kwargs[kde_name][i]
            rvs[pn] = pymc.Deterministic(eval=eval_func_dict[pn],
                                         name=pn,
                                         parents={kde_name:KDE},
                                         doc='model param %s' % pn,
                                         trace=True,
                                         dtype=float,
                                         plot=True)
        return KDE, rvs 


    def _get_limits(self):
        '''
        Determines parameter limits based on prior distribution in self..
        '''
        limits = dict()
        for key, val in self.params.iteritems():
            dist = val[0]
            if dist == 'Uniform':
                limits[key] = [val[1], val[2]]
            elif dist == 'Normal': # use +/- 6 std deviations
                limits[key] = [val[1]-6*np.sqrt(1/val[2])]
            elif dist == 'TruncatedNormal':
                limits[key] = [val[3], val[4]]
            elif dist == 'DiscreteUniform':
                limits[key] = [val[1], val[2]]
            elif dist == 'KDE':
                limits[key] = [None, None]
            else: 
                raise KeyError('The distribution "'+val[0]+'" is not supported')

        return limits


    def _ols(self, qnorm, labels, limits):
        '''
        Least squares function
        '''
        q = self._norm2true(qnorm, labels, limits) 
        f = np.array(self.model.evaluate(q))
        v = np.array(self.data)
        ssq = np.linalg.norm(v-f)**2
        
        print 'DEBUG: this is in MCMC._ols: ssq = %s' % ssq
        return ssq


    def _true2norm(self, q, labels, limits):
        '''
        Normalizes parameters on [0,1]. q is paramMap dictionary. qnorm is 
        list, in alphabetical order by key.
        '''
        qnorm = []
        # step through q dictionary in alphabetical order:
        for key in labels:
            a, b = limits[key]
            qnorm += [(q[key]-a)/(b-a)]
        return qnorm          


    def _norm2true(self, qnorm, labels, limits):
        '''
        Translates parameter values back from normalized [0,1] list to 
        paramMap dictionary.
        '''
        q = dict()
        for i,key in enumerate(labels):
            a, b = limits[key]
            q[key] = qnorm[i]*(b-a)+a
        return q


    def _initialize_plotting(self):
        self.plot_pairwise = self._raise_missing_trace_error
        self.plot_pdf = self._raise_missing_trace_error
        self.plot_residuals = residuals
        self.plot_data = time_vs_observations
        return None


    @staticmethod
    def _raise_missing_trace_error(*args, **kwargs):
        raise ValueError('no trace to plot; use sample() first')


    def _enable_trace_plotting(self):
        trace = self.MCMC.trace
        self.plot_pairwise = functools.partial(pairwise, trace=trace)
        self.plot_pdf = functools.partial(pdf, trace=trace)
        return None



# Helpful functions:

def multivariate_kde_from_samples(prior_pymc_db, param_names):
    '''
    Kernel density estimation of an arbitrary, n-dimensional distribution
    using gaussian kernels. Returns a custom pymc object representing
    this estimated distribution that can be used as a prior for mcmc.

    Only inputs required are a pymc database object containing the param
    markov chain and a list of parameter names to extract from the database.
    A trace will be extracted for each parameter given and a multivariate
    distribution will be fit using KDE to the joint traces.
    '''
    samples = [prior_pymc_db.trace(pn)[:] for pn in param_names]
    samples = np.array(samples)
    return gaussian_kde(samples)


def thin_mcmc_trace(mcmc_filename, thin):
    '''
    Thins an existing pymc chain object with a step size <thin>.
    '''
    mcmc = _load_mcmc_file_as_pickle(mcmc_filename)
    for key in mcmc.keys():
        if key != '_state_':
            mcmc[key][0] = mcmc[key][0][0::thin] # assume 1 trace per key
    save_dir = os.path.split(mcmc_filename)[0]
    _save_thinned_mcmc_obj(mcmc, save_dir)
    return None


def _load_mcmc_file_as_pickle(mcmc_filename):
    try:
        with open(mcmc_filename, 'r') as pf:
            mcmc = pickle.load(pf)
    except IOError:
        raise IOError('mcmc file %s does not exist!' % mcmc_filename)
    return mcmc


def _save_thinned_mcmc_obj(mcmc, working_dir):
    thinned_path = os.path.join(working_dir, 'thinned_mcmc.p')
    with open(thinned_path, 'w') as pf:
        pickle.dump(mcmc, pf)
