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
    pmcmc = ParallelMCMC(model=None, data=None, priors=None, mpi_comm=mock_comm)
    assert pmcmc._size == mock_comm.size
    assert pmcmc._rank == mock_comm.rank


def test_inherit(mock_comm):
    pmcmc = ParallelMCMC(model=None, data=None, priors=None, mpi_comm=mock_comm)
    assert isinstance(pmcmc, MCMCBase)


def test_mpi_eval_model(mocker, mock_comm):
    size = mock_comm.size
    rank = mock_comm.rank

    inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mock_model = lambda x: x

    expected_scatter_input = np.array_split(inputs, size)[rank]
    expected_gather_input = inputs[:expected_scatter_input.shape[0], :2]
    if 0 in expected_gather_input.shape:
        expected_gather_input = np.array([])

    mocker.patch.object(mock_comm, 'allgather', return_value=(inputs[:, :2],))

    pmcmc = ParallelMCMC(mock_model, data=None, priors=None, mpi_comm=mock_comm)
    mocker.patch.object(pmcmc, '_eval_model', side_effect=inputs[:, :2])
    output = pmcmc.evaluate_model(inputs)

    np.testing.assert_array_equal(mock_comm.scatter.call_args[0][0],
                                  expected_scatter_input)
    assert mock_comm.scatter.call_args[1] == {'root': 0}
    np.testing.assert_array_equal(mock_comm.allgather.call_args[0][0],
                                  expected_gather_input)
    np.testing.assert_array_equal(output, inputs[:, :2])
