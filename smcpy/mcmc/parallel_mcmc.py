import numpy as np

from .mcmc_base import MCMCBase
from ..log_likelihoods import Normal


class ParallelMCMC(MCMCBase):
    '''
    Enables use of MPI to split model evaluations over a distributed memory
    system.

    ParallelMCMC is set up such that all positive ranks have effectively
    garbage values. These garbage values serve the purpose of allowing the
    MCMCBase class, which was not written with MPI in mind, to function
    normally and propagate objects of the right shape and type. At the end of
    an analysis that uses the ParallelMCMC class, outputs on positive ranks
    should be discarded and ONLY the output from rank 0 should be used.
    '''

    def __init__(self, model, data, priors, mpi_comm, log_like_args=None,
                 log_like_func=Normal, debug=False):
        self._comm = mpi_comm
        self._size = mpi_comm.Get_size()
        self._rank = mpi_comm.Get_rank()

        super().__init__(model, data, priors, log_like_args, log_like_func,
                         debug)

    def evaluate_model(self, inputs):
        partitioned_inputs = np.array_split(inputs, self._size)
        scattered_inputs = []
        scattered_inputs = self._comm.scatter(partitioned_inputs, root=0)

        scattered_outputs = np.array([]).reshape(0, self._data.size)
        if scattered_inputs.shape[0] > 0:
            scattered_outputs = self._eval_model(scattered_inputs)

        gathered_outputs = self._comm.allgather(scattered_outputs)
        outputs = np.concatenate(gathered_outputs)
        outputs = self._match_output_and_input_shape(outputs, inputs)
        return outputs

    def _match_output_and_input_shape(self, outputs, inputs):
        '''
        Because inputs with zero prior probability are not passed to the
        evaluate_model function of an MCMCBase-derived class, there can be a
        mismatch between nonzero rank inputs.shape and the inputs.shape on
        rank 0. This method corrects for this shape mismatch on return. This is
        another example of why it is important to only use rank 0 outputs.
        '''
        if self._rank == 0:
            return outputs
        return np.zeros((inputs.shape[0], self._data.size))
