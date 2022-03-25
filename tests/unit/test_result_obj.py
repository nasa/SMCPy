import numpy as np
import pytest

from smcpy.utils.storage import *

def test_inmemorystorage_save():
    storage = InMemoryStorage()
    storage.save_step(1)
    storage.save_step(2)
    storage.save_step(3)
    assert storage[0] == 1
    assert storage[1] == 2
    assert storage[2] == 3
    assert all([x == i + 1 for i, x in enumerate(storage)])


@pytest.mark.parametrize('steps', [2, 3, 4, 10])
def test_marginal_log_likelihood_calculation(mocker, steps):
    unnorm_log_weights = np.zeros((5, 1))
    expected_mll = [1] + [5 ** (i + 1) for i in range(steps)]

    mocked_particles = mocker.Mock()
    mocked_particles.compute_total_unnorm_log_wt.return_value = np.log(5)

    storage = InMemoryStorage()
    storage._step_list = [mocked_particles for _ in range(steps)]

    mll = storage.estimate_marginal_log_likelihoods()
    np.testing.assert_array_almost_equal(mll, np.log(expected_mll))


def test_inmemorystorage_cannot_restart():
    storage = InMemoryStorage()
    assert not storage.is_restart


