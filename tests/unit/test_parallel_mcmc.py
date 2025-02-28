import numpy as np
import pytest

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.parallel_vector_mcmc import ParallelVectorMCMC


@pytest.fixture(params=[(1, 0), (2, 1), (3, 1), (4, 3)])
def mock_comm(mocker, request):
    size = request.param[0]
    rank = request.param[1]
    comm = mocker.Mock()
    comm.size = size
    comm.rank = rank
    mocker.patch.object(comm, "Get_size", return_value=size)
    mocker.patch.object(comm, "Get_rank", return_value=rank)
    mocker.patch.object(comm, "scatter")
    return comm


def test_comm_set(mock_comm):
    pmcmc = ParallelVectorMCMC(
        model=None, data=np.ones(4), priors=None, mpi_comm=mock_comm, log_like_args=None
    )
    assert pmcmc._size == mock_comm.size
    assert pmcmc._rank == mock_comm.rank


def test_inherit(mock_comm):
    pmcmc = ParallelVectorMCMC(
        model=None, data=np.ones(4), priors=None, mpi_comm=mock_comm, log_like_args=None
    )
    assert isinstance(pmcmc, VectorMCMC)


def test_mpi_eval_model(mocker, mock_comm):
    size = mock_comm.size
    rank = mock_comm.rank

    inputs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    expected_scatter_input = np.array_split(inputs, size)
    expected_gather_input = expected_scatter_input[rank][:, :2]

    mocker.patch.object(mock_comm, "scatter", return_value=expected_scatter_input[rank])
    mocker.patch.object(
        mock_comm, "allgather", return_value=[y[:, :2] for y in expected_scatter_input]
    )

    mock_model = lambda x: x[:, :2]

    pmcmc = ParallelVectorMCMC(
        mock_model,
        data=inputs[0, :2].flatten(),
        priors=None,
        mpi_comm=mock_comm,
        log_like_args=None,
    )
    output = pmcmc.model_wrapper(mock_model, inputs)

    np.testing.assert_array_equal(
        np.concatenate(mock_comm.scatter.call_args[0][0]),
        np.concatenate(expected_scatter_input),
    )
    np.testing.assert_array_equal(
        mock_comm.allgather.call_args[0][0], expected_gather_input
    )
    assert mock_comm.scatter.call_args[1] == {"root": 0}
    np.testing.assert_array_equal(output, inputs[:, :2])


@pytest.mark.parametrize("data_shape", [(10,), (4, 10)])
def test_mpi_eval_with_zero_scat_input(mocker, mock_comm, data_shape):
    data = np.ones(data_shape)
    inputs = np.ones((100, 3))
    scattered_input = np.array([])
    gathered_outputs = [np.array([1]), np.array([2])]
    mocker.patch.object(mock_comm, "scatter", return_value=scattered_input)
    allgather = mocker.patch.object(
        mock_comm, "allgather", return_value=gathered_outputs
    )

    expected_scattered_output_shape = (0, 10)

    pmcmc = ParallelVectorMCMC(
        mocker.Mock(), data=data, priors=None, mpi_comm=mock_comm, log_like_args=None
    )
    output = pmcmc.model_wrapper(mocker.Mock(), inputs)

    assert allgather.call_args[0][0].shape == expected_scattered_output_shape
