import numpy as np
import pytest

from smcpy.mcmc.parallel_vector_mcmc import ParallelVectorMCMC
from smcpy.utils.single_rank_comm import SingleRankComm


def test_parallel_likelihood(mocker):
    inputs = np.ones((100, 5))
    expected_call = np.ones((1, 100, 5))

    mocked_model_eval = mocker.Mock(return_value=np.array([[1]]))
    mocker.patch(
        "smcpy.mcmc.parallel_vector_mcmc.ParallelVectorMCMC.evaluate_model",
        new=mocked_model_eval,
    )

    mcmc = ParallelVectorMCMC(
        model=None,
        data=np.array([1]),
        priors=None,
        log_like_args=1,
        mpi_comm=SingleRankComm(),
    )
    _ = mcmc.evaluate_log_likelihood(inputs)

    np.testing.assert_array_equal(mocked_model_eval.call_args_list[0][0], expected_call)
