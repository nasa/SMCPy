import numpy as np
import pytest

from smcpy.utils.geweke import compute_geweke


@pytest.mark.parametrize("window", [0, -1, 50.1])
def test_compute_geweke_bad_window(window):
    with pytest.raises(ValueError):
        compute_geweke(None, window)

def test_compute_geweke(mocker):
    window_pct = 10
    window_step = 5
    n_params = 2
    n_samples = 100

    chain = np.ones((n_params, n_samples)) * 2
    chain[:, :int(n_samples / 2)] += 1
    last_x1 = chain[1, 40:int(n_samples / 2)]

    mocked_yw = mocker.Mock(return_value=([1., 1.], np.sqrt(50)))
    mocker.patch("smcpy.utils.geweke.yule_walker", new=mocked_yw)

    expected_burnin = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    expected_z = np.tile(1/np.sqrt(6), (len(expected_burnin), n_params))

    burnin, z = compute_geweke(chain, window_pct=10, step_pct=5)

    np.testing.assert_array_equal(burnin, expected_burnin)
    np.testing.assert_array_almost_equal(z, expected_z)
    assert len(mocked_yw.call_args_list) == (len(z) + 1) * n_params
    np.testing.assert_array_equal(mocked_yw.call_args_list[-1][0][0], last_x1)
