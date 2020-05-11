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

    def __init__(self, vars=None, S=None, proposal_dist=None, scaling=1.,
                 tune=False, tune_interval=100, model=None, mode=None, phi=1,
                 **kwargs):

        self.phi = phi

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(sum(v.dsize for v in vars))

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = MultivariateNormalProposal(S)
        else:
            raise ValueError("Invalid rank for variance: %s" % S.ndim)

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

        posterior_logp = model.logpt - model.observed_RVs[0].logpt + \
                         self.phi * model.observed_RVs[0].logpt
        self.delta_logp = delta_logp(posterior_logp, vars, shared)

        super().__init__(vars, shared)

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

        accept = self.delta_logp(q, q0)
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



