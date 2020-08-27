import numpy as np
import pytest

from smcpy.mcmc.mcmc_base import MCMCBase
from smcpy.mcmc.parallel_mcmc import ParallelMCMC


@pytest.fixture(params=[(1, 0), (2, 1), (3, 1), (4, 3)])
def mock_comm(mocker, request):
    size = request.param[0]
    rank = request.param[1]
    comm = mocker.Mock()
    comm.size = size
    comm.rank = rank
    mocker.patch.object(comm, 'Get_size', return_value=size)
    mocker.patch.object(comm, 'Get_rank', return_value=rank)
    mocker.patch.object(comm, 'scatter')
    return comm

def test_comm_set(mock_comm):
    pmcmc = ParallelMCMC(model=None, data=None, priors=None, std_dev=None,
                         mpi_comm=mock_comm)
    assert pmcmc._size == mock_comm.size
    assert pmcmc._rank == mock_comm.rank


def test_inherit(mock_comm):
    pmcmc = ParallelMCMC(model=None, data=None, priors=None, std_dev=None,
                         mpi_comm=mock_comm)
    assert isinstance(pmcmc, MCMCBase)


def test_mpi_eval_model(mocker, mock_comm):
    size = mock_comm.size
    rank = mock_comm.rank

    inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    expected_scatter_input = np.array_split(inputs, size)
    expected_gather_input = expected_scatter_input[rank][:, :2]

    mocker.patch.object(mock_comm, 'scatter',
                        return_value=expected_scatter_input[rank])
    mocker.patch.object(mock_comm, 'allgather', return_value=\
                        [y[:, :2] for y in expected_scatter_input])

    mock_model = lambda x: x[:, :2]

    pmcmc = ParallelMCMC(mock_model, data=inputs[0, :2].flatten(), priors=None,
                         std_dev=None, mpi_comm=mock_comm)
    output = pmcmc.evaluate_model(inputs)

    np.testing.assert_array_equal(
                              np.concatenate(mock_comm.scatter.call_args[0][0]),
                              np.concatenate(expected_scatter_input))
    np.testing.assert_array_equal(mock_comm.allgather.call_args[0][0],
                                  expected_gather_input)
    assert mock_comm.scatter.call_args[1] == {'root': 0}
    np.testing.assert_array_equal(output, inputs[:, :2])
