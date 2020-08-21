import numpy as np

from mpi4py import MPI

from smcpy.mcmc.parallel_mcmc import ParallelMCMC

def test_parallel_evaluate():
    comm = MPI.COMM_WORLD.Clone()
    rank = comm.Get_rank()
    size = comm.Get_size()

    model = lambda x: x

    inputs = [[i + rank for _ in range(5)] for i in range(size)]
    inputs = np.array(inputs).reshape(-1, 1)
    expected_output = inputs - rank

    pmcmc = ParallelMCMC(model, data=None, priors=None, std_dev=None,
                         mpi_comm=comm)

    np.testing.assert_array_equal(pmcmc.evaluate_model(inputs), expected_output)
