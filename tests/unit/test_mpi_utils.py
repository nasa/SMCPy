import pytest
from smcpy.utils.mpi_utils import *


def test_rank_zero_output_mpi_comm(mocker):
    comm = mocker.Mock()
    comm.bcast.return_value = 0

    class MPIOutputMock:
        def __init__(self, comm):
            self._comm = comm

        @rank_zero_output_only
        def mpi_func(self):
            return 1

    mpi_output_mock = MPIOutputMock(comm)
    res = mpi_output_mock.mpi_func()

    assert res == 0
    # check arguments of bcast
    comm.bcast.assert_called_once_with(1, root=0)


def test_rank_zero_output_kernel_comm(mocker):
    mcmc_kernel = mocker.Mock()
    mcmc_kernel._mcmc._comm.bcast.return_value = 0

    class NoMPIOutputMock:
        def __init__(self, mcmc_kernel):
            self._mcmc_kernel = mcmc_kernel

        @rank_zero_output_only
        def no_mpi_func(self):
            return 1

    no_mpi_output_mock = NoMPIOutputMock(mcmc_kernel)
    res = no_mpi_output_mock.no_mpi_func()

    assert res == 0
    mcmc_kernel._mcmc._comm.bcast.assert_called_once_with(1, root=0)


def test_rank_zero_output_no_comms():
    class NoCommsMock:
        @rank_zero_output_only
        def no_comms_func(self):
            return 1

    no_comms_mock = NoCommsMock()
    res = no_comms_mock.no_comms_func()

    assert res == 1
    with pytest.raises(AttributeError):
        no_comms_mock._comms
    with pytest.raises(AttributeError):
        no_comms_mock._mcmc_kernel


class MPIRunMock:
    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel
        self.func_ran = False

    @rank_zero_run_only
    def mpi_func(self):
        self.func_ran = True


def test_rank_zero_run(mocker):
    mcmc_kernel = mocker.Mock()
    mcmc_kernel._mcmc._rank = 0

    mpi_run_mock = MPIRunMock(mcmc_kernel)
    mpi_run_mock.mpi_func()

    assert mpi_run_mock.func_ran == True
    mcmc_kernel._mcmc._comm.Barrier.assert_called()


@pytest.mark.parametrize("rank", [1, 3, 10, 33, 333])
def test_rank_non_zero_run(mocker, rank):
    mcmc_kernel = mocker.Mock()

    mcmc_kernel._mcmc._rank = rank

    mpi_run_mock = MPIRunMock(mcmc_kernel)
    mpi_run_mock.mpi_func()

    assert mpi_run_mock.func_ran == False
    mcmc_kernel._mcmc._comm.Barrier.assert_called()


def test_no_mpi_zero_run():
    class NoMPIRunMock:
        def __init__(self):
            self.func_ran = False

        @rank_zero_run_only
        def no_mpi_func(self):
            self.func_ran = True

    no_mpi_run_mock = NoMPIRunMock()
    no_mpi_run_mock.no_mpi_func()

    assert no_mpi_run_mock.func_ran == True
    with pytest.raises(AttributeError):
        no_mpi_run_mock._mcmc_kernel
