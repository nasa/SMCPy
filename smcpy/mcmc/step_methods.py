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
import numpy as np
from numpy .random import random
from pymc import StepMethod
from pymc.utils import check_type, round_array, integer_dtypes, bool_dtypes
from pymc.Node import ZeroProbability, Variable
import warnings

class DelayedRejectionAdaptiveMetropolis(StepMethod):

    """
    The DelayedRejectionAdaptativeMetropolis (DRAM) sampling algorithm works
    like AdaptiveMetropolis, except that if a bold initial jump proposal is
    rejected, a more conservative jump proposal will be tried. Although the
    chain is non-Markovian, as with AdaptiveMetropolis, it too has correct
    ergodic properties. See (Haario et al., 2006) for details.

    :Parameters:
      - stochastic : PyMC objects
          Stochastic objects to be handled by the AM algorith,

      - cov : array
          Initial guess for the covariance matrix C. If it is None, the
          covariance will be estimated using the scales dictionary if provided,
          the existing trace if available, or the current stochastics value.
          It is suggested to provide a sensible guess for the covariance, and
          not rely on the automatic assignment from stochastics value.

      - delay : int
          Number of steps before the empirical covariance is computed. If greedy
          is True, the algorithm waits for delay *accepted* steps before
          computing the covariance.

      - interval : int
          Interval between covariance updates. Higher dimensional spaces require
          more samples to obtain reliable estimates for the covariance updates.

      - greedy : bool
          If True, only the accepted jumps are tallied in the internal trace
          until delay is reached. This is useful to make sure that the empirical
          covariance has a sensible structure.

      - shrink_if_necessary : bool
          If True, the acceptance rate is checked when the step method tunes. If
          the acceptance rate is small, the proposal covariance is shrunk 
          according to the following rule:

          if acc_rate < .001:
              self.C *= .01
          elif acc_rate < .01:
              self.C *= .25

      - scales : dict
          Dictionary containing the scale for each stochastic keyed by name.
          If cov is None, those scales are used to define an initial covariance
          matrix. If neither cov nor scale is given, the initial covariance is
          guessed from the trace (it if exists) or the objects value, alt

      - verbose : int
          Controls the verbosity level.


    :Notes:
    Use the methods: `cov_from_scales`, `cov_from_trace` and `cov_from_values`
    for more control on the creation of an initial covariance matrix. A lot of
    problems can be avoided with a good initial covariance and long enough 
    intervals between covariance updates. That is, do not compensate for a bad
    covariance guess by reducing the interval between updates thinking the
    covariance matrix will converge more rapidly.


    :Reference:
      Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm,
          Bernouilli, vol. 7 (2), pp. 223-242, 2001.
      Haario 2006.
    """

    def __init__(self, stochastic, cov=None, delay=1000, interval=200,
                 greedy=True, drscale = 0.1, shrink_if_necessary=False,
                 scales=None, verbose=-1, tally=False):

        # Verbosity flag
        self.verbose = verbose

        self.accepted = 0
        self.rejected_then_accepted = 0
        self.rejected_twice = 0

        if not np.iterable(stochastic) or isinstance(stochastic, Variable):
            stochastic = [stochastic]


        # Initialize superclass
        StepMethod.__init__(self, stochastic, verbose, tally)

        self._id = 'DelayedRejectionAdaptiveMetropolis_' + '_'.join(
            [p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += [
            'accepted', 'rejected_then_accepted','rejected_twice', 
            '_trace_count', '_current_iter', 'C', 'proposal_sd',
            '_proposal_deviate', '_trace', 'shrink_if_necessary']
        self._tuning_info = ['C']

        self.proposal_sd = None
        self.shrink_if_necessary = shrink_if_necessary

        # Number of successful steps before the empirical covariance is
        # computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy
        # Scale for second attempt
        self.drscale = drscale

        # Initialization methods
        self.check_type()
        self.dimension()

        # Set the initial covariance using cov, or the following fallback
        # mechanisms:
        # 1. If scales is provided, use it.
        # 2. If a trace is present, compute the covariance matrix empirically
        #    from it.
        # 3. Use the stochastics value as a guess of the variance.
        if cov is not None:
            self.C = cov
        elif scales:
            self.C = self.cov_from_scales(scales)
        else:
            try:
                self.C = self.cov_from_trace()
            except AttributeError:
                self.C = self.cov_from_value(100.)

        self.updateproposal_sd()

        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy
        # sampling can be done during warm-up period.
        self._trace_count = 0
        self._current_iter = 0

        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []

        if self.verbose >= 2:
            print("Initialization...")
            print('Dimension: ', self.dim)
            print("C_0: ", self.C)
            print("Sigma: ", self.proposal_sd)

    @staticmethod
    def competence(stochastic):
        """
        DRAM must be applied manually via MCMC.use_step_method().
        """
        return 0

    def cov_from_value(self, scaling):
        """Return a covariance matrix for the jump distribution using
        the actual value of the stochastic as a guess of their variance,
        divided by the `scaling` argument.

        Note that this is likely to return a poor guess.
        """
        rv = []
        for s in self.stochastics:
            rv.extend(np.ravel(s.value).copy())

        # Remove 0 values since this would lead to quite small jumps...
        arv = np.array(rv)
        arv[arv == 0] = 1.

        # Create a diagonal covariance matrix using the scaling factor.
        return np.eye(self.dim) * np.abs(arv) / scaling

    def cov_from_scales(self, scales):
        """Return a covariance matrix built from a dictionary of scales.

        `scales` is a dictionary keyed by stochastic instances, and the
        values refer are the variance of the jump distribution for each
        stochastic. If a stochastic is a sequence, the variance must
        have the same length.
        """

        # Get array of scales
        ord_sc = []
        for stochastic in self.stochastics:
            ord_sc.append(np.ravel(scales[stochastic]))
        ord_sc = np.concatenate(ord_sc)

        if np.squeeze(ord_sc).shape[0] != self.dim:
            raise ValueError("Improper initial scales, dimension don't match",
                             (np.squeeze(ord_sc), self.dim))

        # Scale identity matrix
        return np.eye(self.dim) * ord_sc

    def cov_from_trace(self, trace=slice(None)):
        """Define the jump distribution covariance matrix from the object's
        stored trace.

        :Parameters:
        - `trace` : slice or int
          A slice for the stochastic object's trace in the last chain, or a
          an integer indicating the how many of the last samples will be used.

        """
        n = []
        for s in self.stochastics:
            n.append(s.trace.length())
        n = set(n)
        if len(n) > 1:
            raise ValueError('Traces do not have the same length.')
        elif n == 0:
            raise AttributeError(
                'Stochastic has no trace to compute covariance.')
        else:
            n = n.pop()

        if not isinstance(trace, slice):
            trace = slice(trace, n)

        a = self.trace2array(trace)

        return np.cov(a, rowvar=0)

    def check_type(self):
        """Make sure each stochastic has a correct type, and identify discrete
        stochastics."""
        self.isdiscrete = {}
        for stochastic in self.stochastics:
            if stochastic.dtype in integer_dtypes:
                self.isdiscrete[stochastic] = True
            elif stochastic.dtype in bool_dtypes:
                raise ValueError(
                    'Binary stochastics not supported by AdaptativeMetropolis.')
            else:
                self.isdiscrete[stochastic] = False

    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stochastic.
        """
        self.dim = 0
        self._slices = {}
        for stochastic in self.stochastics:
            if isinstance(stochastic.value, np.matrix):
                p_len = len(stochastic.value.A.ravel())
            elif isinstance(stochastic.value, np.ndarray):
                p_len = len(stochastic.value.ravel())
            else:
                p_len = 1
            self._slices[stochastic] = slice(self.dim, self.dim + p_len)
            self.dim += p_len

    def update_cov(self):
        """Recursively compute the covariance matrix for the multivariate normal
        proposal distribution.

        This method is called every self.interval once self.delay iterations
        have been performed.
        """

        scaling = (2.4) ** 2 / self.dim  # Gelman et al. 1996.
        epsilon = 1.0e-5
        chain = np.asarray(self._trace)

        # Recursively compute the chain mean
        self.C, self.chain_mean = self.recursive_cov(self.C, self._trace_count,
                                                     self.chain_mean, chain,
                                                     scaling=scaling,
                                                     epsilon=epsilon)

        # Shrink covariance if acceptance rate is way too small
        acc_rate = (self.accepted + self.rejected_then_accepted) / \
                   (self.accepted + 2.*self.rejected_then_accepted + \
                    2.*self.rejected_twice)
        if self.shrink_if_necessary:
            if acc_rate < .001:
                self.C *= .01
            elif acc_rate < .01:
                self.C *= .25
            if self.verbose > 1:
                if acc_rate < .01:
                    print(
                        '\tAcceptance rate was',
                        acc_rate,
                        'shrinking covariance')
        self.accepted = 0.
        self.rejected_then_accepted = 0.
        self.rejected_twice = 0.

        if self.verbose > 1:
            print("\tUpdating covariance ...\n", self.C)
            print("\tUpdating mean ... ", self.chain_mean)

        # Update state
        adjustmentwarning = '\n' +\
            'Covariance was not positive definite and proposal_sd cannot be computed by \n' + \
            'Cholesky decomposition. The next jumps will be based on the last \n' + \
            'valid covariance matrix. This situation may have arisen because no \n' + \
            'jumps were accepted during the last `interval`. One solution is to \n' + \
            'increase the interval, or specify an initial covariance matrix with \n' + \
            'a smaller variance. For this simulation, each time a similar error \n' + \
            'occurs, proposal_sd will be reduced by a factor .9 to reduce the \n' + \
            'jumps and increase the likelihood of accepted jumps.'

        try:
            self.updateproposal_sd()
        except np.linalg.LinAlgError:
            warnings.warn(adjustmentwarning)
            #self.covariance_adjustment(.9)

        self._trace_count += len(self._trace)
        self._trace = []

    def covariance_adjustment(self, f=.9):
        """Multiply self.proposal_sd by a factor f. This is useful when the current proposal_sd is too large and all jumps are rejected.
        """
        self.proposal_sd *= f
        self.proposal_sd_inv = np.linalg.inv(self.proposal_sd)

    def updateproposal_sd(self):
        """Compute the Cholesky decomposition of self.C."""
        self.proposal_sd = np.linalg.cholesky(self.C)
        self.proposal_sd_inv = np.linalg.inv(self.proposal_sd)

    def recursive_cov(self, cov, length, mean, chain, scaling=1, epsilon=0):
        r"""Compute the covariance recursively.

        Return the new covariance and the new mean.

        .. math::
            C_k & = \frac{1}{k-1} (\sum_{i=1}^k x_i x_i^T - k\bar{x_k}\bar{x_k}^T)
            C_n & = \frac{1}{n-1} (\sum_{i=1}^k x_i x_i^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)
                & = \frac{1}{n-1} ((k-1)C_k + k\bar{x_k}\bar{x_k}^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)

        :Parameters:
            -  cov : matrix
                Previous covariance matrix.
            -  length : int
                Length of chain used to compute the previous covariance.
            -  mean : array
                Previous mean.
            -  chain : array
                Sample used to update covariance.
            -  scaling : float
                Scaling parameter
            -  epsilon : float
                Set to a small value to avoid singular matrices.
        """
        n = length + len(chain)
        k = length
        new_mean = self.recursive_mean(mean, length, chain)

        t0 = k * np.outer(mean, mean)
        t1 = np.dot(chain.T, chain)
        t2 = n * np.outer(new_mean, new_mean)
        t3 = epsilon * np.eye(cov.shape[0])

        new_cov = (k - 1) / (n - 1.) * cov + scaling / (n - 1.) * (t0 + t1 - t2 + t3)
        return new_cov, new_mean

    def recursive_mean(self, mean, length, chain):
        r"""Compute the chain mean recursively.

        Instead of computing the mean :math:`\bar{x_n}` of the entire chain,
        use the last computed mean :math:`bar{x_j}` and the tail of the chain
        to recursively estimate the mean.

        .. math::
            \bar{x_n} & = \frac{1}{n} \sum_{i=1}^n x_i
                      & = \frac{1}{n} (\sum_{i=1}^j x_i + \sum_{i=j+1}^n x_i)
                      & = \frac{j\bar{x_j}}{n} + \frac{\sum_{i=j+1}^n x_i}{n}

        :Parameters:
            -  mean : array
                Previous mean.
            -  length : int
                Length of chain used to compute the previous mean.
            -  chain : array
                Sample used to update mean.
        """
        n = length + len(chain)
        return length * mean / n + chain.sum(0) / n

    def propose_first(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.

        The proposal jumps are drawn from a multivariate normal distribution.
        """

        arrayjump = np.dot( self.proposal_sd,\
                            np.random.normal(size=self.proposal_sd.shape[0]))
        # save in case needed for calculating second proposal probability
        self.arrayjump1 = arrayjump[:]

        if self.verbose > 2: print('First jump:', arrayjump)

        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]].squeeze()
            if np.iterable(stochastic.value):
                jump = np.reshape( arrayjump[ self._slices[stochastic]],\
                                                 np.shape(stochastic.value))
            if self.isdiscrete[stochastic]:
                jump = round_array(jump)
            stochastic.value = stochastic.value + jump

    def propose_second(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.

        The proposal jumps are drawn from a multivariate normal distribution.
        """

        arrayjump = self.drscale*np.dot( self.proposal_sd,\
                            np.random.normal(size=self.proposal_sd.shape[0]))

        if self.verbose > 2: print('Second jump:', arrayjump)

        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]].squeeze()
            if np.iterable(stochastic.value):
                jump = np.reshape( arrayjump[ self._slices[stochastic]],\
                                                 np.shape(stochastic.value))
            if self.isdiscrete[stochastic]:
                jump = round_array(jump)
            stochastic.value = stochastic.value + jump

        arrayjump1 = self.arrayjump1
        arrayjump2 = arrayjump

        self.q = np.exp(-0.5*(
                np.linalg.norm(np.dot(arrayjump2-arrayjump1,self.proposal_sd_inv),ord=2)**2 -
                np.linalg.norm(np.dot(-arrayjump1, self.proposal_sd_inv),ord=2)**2 ))

    def step(self):
        """
        Perform a Metropolis step.

        Stochastic parameters are block-updated using a multivariate normal
        distribution whose covariance is updated every self.interval once
        self.delay steps have been performed.

        The AM instance keeps a local copy of the stochastic parameter's trace.
        This trace is used to computed the empirical covariance, and is
        completely independent from the Database backend.

        If self.greedy is True and the number of iterations is smaller than
        self.delay, only accepted jumps are stored in the internal
        trace to avoid computing singular covariance matrices.
        """

        # Probability and likelihood for stochastic's current value:
        # PROBLEM: I am using logp plus loglike everywhere... and I shouldn't be!!
        logp = self.logp_plus_loglike
        if self.verbose > 1:
            print('Current value: ', self.stoch2array())
            print('Current likelihood: ', logp)

        # Sample a candidate value
        self.propose_first()

        # Metropolis acception/rejection test
        accept = False
        try:
            # Probability and likelihood for stochastic's 1st proposed value:
            self.logp_p1 = self.logp_plus_loglike
            logp_p1 = float(self.logp_p1)
            self.logalpha01 = min(0, logp_p1 - logp)
            logalpha01 = self.logalpha01
            if self.verbose > 2:
                print('First proposed value: ', self.stoch2array())
                print('First proposed likelihood: ', logp_p1)

            if np.log(random()) < logalpha01:
                accept = True
                self.accepted += 1
                if self.verbose > 2:
                    print('Accepted')
                logp_p = logp_p1
            else:
                if self.verbose > 2:
                    print('Delaying rejection...')
                for stochastic in self.stochastics:
                    stochastic.revert()
                self.propose_second()
                try:
                    # Probability and likelihood for stochastic's 2nd proposed
                    # value:
                    # CHECK THAT THIS IS RECALCULATED WITH PROPOSE_SECODN
                    # CHECK THAT logp_p1 iS NOT CHANGED WHEN THIS IS RECALCD
                    logp_p2 = self.logp_plus_loglike
                    logalpha21 = min(0, logp_p1 - logp_p2)
                    l = logp_p2 - logp
                    q = self.q
                    logalpha_02 = np.log(l*q*(1-np.exp(logalpha21))/(1-np.exp(logalpha01)))
                    if self.verbose > 2:
                        print('Second proposed value: ', self.stoch2array())
                        print('Second proposed likelihood: ', logp_p2)
                    if np.log(random()) < logalpha_02:
                        accept = True
                        self.rejected_then_accepted += 1
                        logp_p = logp_p2
                        if self.verbose > 2:
                            print('Accepted after one rejection')
                    else:
                        self.rejected_twice += 1
                        logp_p = None
                        if self.verbose > 2:
                            print('Rejected twice')
                except ZeroProbability:
                    self.rejected_twice += 1
                    logp_p = None
                    if self.verbose > 2:
                        print('Rejected twice')
        except ZeroProbability:
            if self.verbose > 2:
                print('Delaying rejection...')
            for stochastic in self.stochastics:
                stochastic.revert()
            self.propose_second()
            try:
                # Probability and likelihood for stochastic's proposed value:
                logp_p2 = self.logp_plus_loglike
                logp_p1 = -np.inf
                logalpha01 = -np.inf
                logalpha21 = min(0, logp_p1 - logp_p2)
                l = np.exp(logp_p2 - logp)
                q = self.q
                logalpha_02 = np.log(l*q*(1-np.exp(logalpha21))/(1-np.exp(logalpha01)))
                if self.verbose > 2:
                    print('Second proposed value: ', self.stoch2array())
                    print('Second proposed likelihood: ', logp_p2)
                if np.log(random()) < logalpha_02:
                    accept = True
                    self.rejected_then_accepted += 1
                    logp_p = logp_p2
                    if self.verbose > 2:
                        print('Accepted after one rejection with ZeroProbability')
                else:
                    self.rejected_twice += 1
                    logp_p = None
                    if self.verbose > 2:
                        print('Rejected twice')
            except ZeroProbability:
                self.rejected_twice += 1
                logp_p = None
                if self.verbose > 2:
                    print('Rejected twice with ZeroProbability Error.')
        #print('\n\nRejected then accepted number of times: ',self.rejected_then_accepted)
        #print('Rejected twice number of times: ',self.rejected_twice)

        if (not self._current_iter % self.interval) and self.verbose > 1:
            print("Step ", self._current_iter)
            print("\tLogprobability (current, proposed): ", logp, logp_p)
            for stochastic in self.stochastics:
                print(
                    "\t",
                    stochastic.__name__,
                    stochastic.last_value,
                    stochastic.value)
            if accept:
                print("\tAccepted\t*******\n")
            else:
                print("\tRejected\n")
            print(
                "\tAcceptance ratio: ",
                (self.accepted + self.rejected_then_accepted) / (
                    self.accepted + 2.*self.rejected_then_accepted + 2.*self.rejected_twice))

        if self._current_iter == self.delay:
            self.greedy = False

        if not accept:
            self.reject()

        if accept or not self.greedy:
            self.internal_tally()

        if self._current_iter > self.delay and self._current_iter % self.interval == 0:
            self.update_cov()

        self._current_iter += 1

    # Please keep reject() factored out- helps RandomRealizations figure out
    # what to do.
    def reject(self):
        for stochastic in self.stochastics:
            # stochastic.value = stochastic.last_value
            stochastic.revert()

    def internal_tally(self):
        """Store the trace of stochastics for the computation of the covariance.
        This trace is completely independent from the backend used by the
        sampler to store the samples."""
        chain = []
        for stochastic in self.stochastics:
            chain.append(np.ravel(stochastic.value))
        self._trace.append(np.concatenate(chain))

    def trace2array(self, sl):
        """Return an array with the trace of all stochastics, sliced by sl."""
        chain = []
        for stochastic in self.stochastics:
            tr = stochastic.trace.gettrace(slicing=sl)
            if tr is None:
                raise AttributeError
            chain.append(tr)
        return np.hstack(chain)

    def stoch2array(self):
        """Return the stochastic objects as an array."""
        a = np.empty(self.dim)
        for stochastic in self.stochastics:
            a[self._slices[stochastic]] = stochastic.value
        return a

    def tune(self, verbose=0):
        """Tuning is done during the entire run, independently from the Sampler
        tuning specifications. """
        return False


class SMC_Metropolis(StepMethod):

    """
    The AdaptativeMetropolis (AM) sampling algorithm works like a regular
    Metropolis, with the exception that stochastic parameters are block-updated
    using a multivariate jump distribution whose covariance is tuned during
    sampling. Although the chain is non-Markovian, i.e. the proposal
    distribution is asymmetric, it has correct ergodic properties. See
    (Haario et al., 2001) for details.

    :Parameters:
      - stochastic : PyMC objects
          Stochastic objects to be handled by the AM algorith,

      - cov : array
          Initial guess for the covariance matrix C. If it is None, the
          covariance will be estimated using the scales dictionary if provided,
          the existing trace if available, or the current stochastics value.
          It is suggested to provide a sensible guess for the covariance, and
          not rely on the automatic assignment from stochastics value.

      - shrink_if_necessary : bool
          If True, the acceptance rate is checked when the step method tunes. If
          the acceptance rate is small, the proposal covariance is shrunk according
          to the following rule:

          if acc_rate < .001:
              self.C *= .01
          elif acc_rate < .01:
              self.C *= .25

      - scales : dict
          Dictionary containing the scale for each stochastic keyed by name.
          If cov is None, those scales are used to define an initial covariance
          matrix. If neither cov nor scale is given, the initial covariance is
          guessed from the trace (it if exists) or the objects value, alt

      - verbose : int
          Controls the verbosity level.

    :Reference:
      Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm,
          Bernouilli, vol. 7 (2), pp. 223-242, 2001.
    """

    def __init__(self, stochastic, cov, phi, verbose=-1, tally=False):
        # assign cooling step
        self.phi = phi

        # Verbosity flag
        self.verbose = verbose

        self.accepted = 0
        self.rejected = 0

        if not np.iterable(stochastic) or isinstance(stochastic, Variable):
            stochastic = [stochastic]

        # Initialize superclass
        StepMethod.__init__(self, stochastic, verbose, tally)

        self._id = 'SMC_Metropolis_' + '_'.join(
            [p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += [
            'accepted', 'rejected', '_trace_count', '_current_iter', 'C',
            'proposal_sd', '_proposal_deviate', '_trace', 'shrink_if_necessary']
        self._tuning_info = ['C']

        self.proposal_sd = None
        self.shrink_if_necessary = False

        # Initialization methods
        self.check_type()
        self.dimension()

        # Set the initial covariance using cov
        self.C = cov
        self.updateproposal_sd()

        # Keep track of the internal trace length
        self._trace_count = 0
        self._current_iter = 0

        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []

        if self.verbose >= 2:
            print("Initialization...")
            print('Dimension: ', self.dim)
            print("C_0: ", self.C)
            print("Sigma: ", self.proposal_sd)

    @staticmethod
    def competence(stochastic):
        """
        The competence function for AdaptiveMetropolis.
        The AM algorithm is well suited to deal with multivariate
        parameters, particularly those which are correlated with one
        another. However, it does not work reliably with all multivariate
        stochastics, so it must be applied manually via MCMC.use_step_method().
        """
        return 0


    def check_type(self):
        """Make sure each stochastic has a correct type, and identify discrete
           stochastics."""
        self.isdiscrete = {}
        for stochastic in self.stochastics:
            if stochastic.dtype in integer_dtypes:
                self.isdiscrete[stochastic] = True
            elif stochastic.dtype in bool_dtypes:
                raise ValueError(
                    'Binary stochastics not supported by AdaptativeMetropolis.')
            else:
                self.isdiscrete[stochastic] = False

    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stochastic.
        """
        self.dim = 0
        self._slices = {}
        for stochastic in self.stochastics:
            if isinstance(stochastic.value, np.matrix):
                p_len = len(stochastic.value.A.ravel())
            elif isinstance(stochastic.value, np.ndarray):
                p_len = len(stochastic.value.ravel())
            else:
                p_len = 1
            self._slices[stochastic] = slice(self.dim, self.dim + p_len)
            self.dim += p_len

    def updateproposal_sd(self):
        """Compute the Cholesky decomposition of self.C."""
        self.proposal_sd = np.linalg.cholesky(self.C)


    def propose(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.

        The proposal jumps are drawn from a multivariate normal distribution.
        """

        arrayjump = np.dot(
            self.proposal_sd,
            np.random.normal(
                size=self.proposal_sd.shape[
                    0]))
        if self.verbose > 2:
            print('Jump :', arrayjump)

        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]].squeeze()
            if np.iterable(stochastic.value):
                jump = np.reshape(
                    arrayjump[
                        self._slices[
                            stochastic]],
                    np.shape(
                        stochastic.value))
            if self.isdiscrete[stochastic]:
                jump = round_array(jump)
            stochastic.value = stochastic.value + jump

    def step(self):
        """
        Perform a Metropolis step.

        Stochastic parameters are block-updated using a multivariate normal
        distribution with covariance self.C.
        """

        # Probability and likelihood for stochastic's current value:
        logp = get_gamma_smc(self.stochastics, self.children, self.phi)
        if self.verbose > 1:
            print('Current value: ', self.stoch2array())
            print('Current likelihood: ', logp)

        # Sample a candidate value
        self.propose()

        # Metropolis acception/rejection test
        accept = False
        try:
            # Probability and likelihood for stochastic's proposed value:
            logp_p = get_gamma_smc(self.stochastics, self.children, self.phi)
            if self.verbose > 2:
                print('Current value: ', self.stoch2array())
                print('Current likelihood: ', logp_p)

            if np.log(random()) < logp_p - logp:
                accept = True
                self.accepted += 1
                if self.verbose > 2:
                    print('Accepted')
            else:
                self.rejected += 1
                if self.verbose > 2:
                    print('Rejected')
        except ZeroProbability:
            self.rejected += 1
            logp_p = None
            if self.verbose > 2:
                print('Rejected with ZeroProbability Error.')

        if self.verbose > 1:
            print("Step ", self._current_iter)
            print("\tLogprobability (current, proposed): ", logp, logp_p)
            for stochastic in self.stochastics:
                print(
                    "\t",
                    stochastic.__name__,
                    stochastic.last_value,
                    stochastic.value)
            if accept:
                print("\tAccepted\t*******\n")
            else:
                print("\tRejected\n")
            print(
                "\tAcceptance ratio: ",
                self.accepted / (
                    self.accepted + self.rejected))

        if not accept:
            self.reject()

        self._current_iter += 1

    # Please keep reject() factored out- helps RandomRealizations figure out
    # what to do.
    def reject(self):
        for stochastic in self.stochastics:
            # stochastic.value = stochastic.last_value
            stochastic.revert()

    def stoch2array(self):
        """Return the stochastic objects as an array."""
        a = np.empty(self.dim)
        for stochastic in self.stochastics:
            a[self._slices[stochastic]] = stochastic.value
        return a

    def tune(self, verbose=0):
        """Tuning is done during the entire run, independently from the Sampler
        tuning specifications. """
        return False


def get_gamma_smc(stochastics, children, phi):
    '''
    Computes p(theta)*p(y|theta)**phi given the parameters theta (stochastics)
    and the results object (children)
    '''
    logp = []
    for s in list(stochastics):
        logp.append(s.logp)
    log_like = list(children)[0].logp
    return np.sum(logp)+phi*log_like
