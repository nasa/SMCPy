from smcpy.utils.single_rank_comm import SingleRankComm
import numpy as np
import pytest

array_equal = np.testing.assert_array_equal


@pytest.fixture
def single_rank_comm():
    return SingleRankComm()


def test_get_rank(single_rank_comm):
    assert single_rank_comm.Get_rank() == 0


def test_get_size(single_rank_comm):
    assert single_rank_comm.Get_size() == 1


def test_scatter_when_length_scatter_list_one(single_rank_comm):
    scatter_list = [np.array([0, 1, 2, 3])]
    array_equal(single_rank_comm.scatter(scatter_list), np.array([0, 1, 2, 3]))


def test_raise_error_when_length_scatter_list_not_one(single_rank_comm):
    scatter_list = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError):
        single_rank_comm.scatter(scatter_list)


def test_gather(single_rank_comm):
    scatter_list = np.array([0, 1, 2, 3])
    array_equal(single_rank_comm.gather(scatter_list), [scatter_list])
