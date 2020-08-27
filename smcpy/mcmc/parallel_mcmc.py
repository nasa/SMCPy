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
        scattered_inputs = []
        scattered_inputs = self._comm.scatter(partitioned_inputs, root=0)

        scattered_outputs = np.array([]).reshape(0, self._data.size)
        if scattered_inputs.shape[0] > 0:
            scattered_outputs = self._eval_model(scattered_inputs)

        gathered_outputs = self._comm.allgather(scattered_outputs)
        return np.concatenate(gathered_outputs)
