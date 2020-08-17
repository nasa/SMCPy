def rank_zero_output_only(func):
    """
    Wrapper function that detects whether mpi4py is available and forces all
    ranks to output from rank 0 only. Intended to be used as a decorator where
    mpi-enabled MCMC kernels are deployed.

    Example use case: in the built-in mpi implementation of MCMC
    (ParallelMCMC), the likelihoods are synced but particle parameter values
    are not, meaning that ranks other than rank 0 essentially carry dummy (and
    fundamentally incorrect) Particles objects. This wrapper ensures that only
    rank 0 Particles objects are returned to the user. Note that this would not
    be required if a global random seed is set prior to execution of any SMCPy
    parallel code.
    """
    def wrapper(self, *args, **kwargs):
        output = func(self, *args, **kwargs)

        try:
            output = self._mcmc_kernel._mcmc._comm.bcast(output, root=0)
        except AttributeError:
            pass

        return output

    return wrapper
