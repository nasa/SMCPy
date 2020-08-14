import numpy as np

from .mcmc_base import MCMCBase


class ParallelMCMC(MCMCBase):

    def __init__(self, model, data, priors, std_dev, mpi_comm):
        self._comm = mpi_comm
        self._size = mpi_comm.Get_size()
        self._rank = mpi_comm.Get_rank()

        super().__init__(model, data, priors, std_dev)

    def evaluate_model(self, inputs):
        partitioned_inputs = np.array_split(inputs, self._size)
        inputs = self._comm.scatter(partitioned_inputs, root=0)
        outputs = self._eval_model(inputs)
        return np.concatenate(self._comm.allgather(outputs))
