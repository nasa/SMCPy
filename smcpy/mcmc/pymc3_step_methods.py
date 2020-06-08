import abc
import numpy as np
import pymc3 as pm

from pymc3.step_methods.metropolis import *
from pymc3.step_methods.metropolis import delta_logp
from pymc3.step_methods.arraystep import ArrayStepShared, metrop_select
from pymc3.step_methods.arraystep import Competence
from pymc3.theanof import floatX

from smcpy.mcmc.step_method_base import SMCStepMethod


class SMCMetropolis(ArrayStepShared, SMCStepMethod):
    """
    Modification of the PyMC3 Metropolis step method for use with SMCPy; 
    main changes are to allow for manually changing the proposal distribution
    standard deviation or covariance on the fly, and for specifying a phi value
    for tempering the likelihood function. Original PyMC3 docstring is copied
    below.

    ---

    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    S : standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal.
    scaling : scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune : bool
        Flag for tuning. Defaults to False.
    tune_interval : int
        The frequency of tuning. Defaults to 100 iterations.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    mode :  string or `Mode` instance.
        compilation mode passed to Theano functions
    """
    name = 'smc_metropolis'

    default_blocked = False
    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'tune': np.bool,
    }]

    def __init__(self, vars=None, proposal_dist=None, scaling=1.,
                 tune=False, tune_interval=100, model=None, mode=None,
                 **kwargs):

        self._proposal_dist = proposal_dist

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)
        self._vars = vars

        self.S = np.ones(sum(v.dsize for v in vars))

        self.scaling = np.atleast_1d(scaling).astype('d')
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0

        # Determine type of variables
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        self.mode = mode

        shared = pm.make_shared_replacements(vars, model)

        prior_logp = model.logpt - model.observed_RVs[0].logpt
        likelihood_logp = model.observed_RVs[0].logpt
        posterior_logp = model.logpt

        self.delta_prior_logp = delta_logp(prior_logp, vars, shared)
        #self.delta_like_logp = delta_logp(likelihood_logp, vars, shared)
        self.delta_post_logp = delta_logp(posterior_logp, vars, shared)

        super().__init__(vars, shared)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi <= 1.0 and phi >= 0.0:
            self._phi = phi
        else:
            raise ValueError('phi = {} not in [0, 1]'.format(phi))

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, S):
        if self._proposal_dist is not None:
            self.proposal_dist = self._proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = MultivariateNormalProposal(S)
        else:
            raise ValueError("Invalid rank for variance: %s" % S.ndim)
        self._S = S

    def calc_acceptance_ratio(self, q, q0):
        likelihood = self.delta_post_logp(q, q0) - self.delta_prior_logp(q, q0)
        return likelihood * self.phi + self.delta_prior_logp(q, q0)

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype('int64')
                q0 = q0.astype('int64')
                q = (q0 + delta).astype('int64')
            else:
                delta[self.discrete] = np.round(
                    delta[self.discrete], 0)
                q = (q0 + delta)
        else:
            q = floatX(q0 + delta)

        accept = self.calc_acceptance_ratio(q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            'tune': self.tune,
            'accept': np.exp(accept),
        }

        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE
