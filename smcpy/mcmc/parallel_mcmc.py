import numpy as np

from .mcmc_base import MCMCBase


class ParallelMCMC(MCMCBase):

    def __init__(self, model, data, priors, mpi_comm):
        self._comm = mpi_comm
        self._size = mpi_comm.Get_size()
        self._rank = mpi_comm.Get_rank()

        super().__init__(model, data, priors)

    def evaluate_model(self, inputs):
        partitioned_inputs = np.array_split(inputs, self._size)[self._rank]
        self._comm.scatter(partitioned_inputs, root=0)
        partitioned_outputs = []
        for input_ in partitioned_inputs:
            partitioned_outputs.append(self._eval_model(input_))
        return self._comm.gather(np.array(partitioned_outputs), root=0)
