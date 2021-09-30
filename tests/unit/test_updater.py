import numpy as np
import pytest

from smcpy.smc.updater import Updater


@pytest.mark.parametrize('ess_threshold', [-0.1, 1.1])
def test_ess_threshold_valid(ess_threshold):
    with pytest.raises(ValueError):
        Updater(ess_threshold)


def test_update(mocker):
    p_mock = mocker.Mock()
    p_class_mock = mocker.patch('smcpy.smc.updater.Particles', return_value=0)
    updater = Updater(0.5)
    updater._compute_new_weights = mocker.Mock()
    updater.resample_if_needed = mocker.Mock()

    updater(p_mock, delta_phi=2)

    updater.resample_if_needed.assert_called_once_with(0)
    p_class_mock.called_once_with(p_mock.param_dict, p_mock.log_likes, 1)
    updater._compute_new_weights.assert_called_once_with(p_mock, 2)


@pytest.mark.parametrize('u_samples, resample_indices',
                         ((np.array([[[.3], [.3], [.7]]]), [1, 1, 2]),
                          (np.array([.3, .05, .05]), [1, 0, 0]),
                          (np.array([.3, .05, .7]), [1, 0, 2]),
                          (np.array([.05, .05, .7]), [0, 0, 2]),
                          (np.array([.3, .7, .3]), [1, 2, 1])))
def test_update_with_resample(u_samples, resample_indices, mocker):
    u_samples = np.tile(u_samples, (2, 1, 1))
    mocker.patch('smcpy.smc.updater.np.random.uniform', return_value=u_samples)
    particle_obj = mocker.patch('smcpy.smc.updater.Particles')
    updater = Updater(ess_threshold=1.0)


    params = np.arange(12).reshape(2, 3, 2)
    weights = [0.1, 0.4, 0.5]
    log_likes = np.ones((2, 3, 1))
    names = ['a', 'b']
    particles = mocker.Mock()
    particles.params = params
    particles.param_names = names
    particles.log_likes = log_likes
    particles.weights = np.transpose(np.tile(weights, (2, 1, 1)), (0, 2, 1))
    particles.num_particles = 3
    particles.compute_ess.return_value = 0

    expected_params = params[:, resample_indices, :]
    expected_log_likes = log_likes[:, resample_indices, :]
    expected_log_weights = np.log(np.full((2, 3, 1), 1/3))

    _ = updater.resample_if_needed(particles)

    param_dict, log_like, wts = particle_obj.call_args_list[0][0]
    
    np.testing.assert_array_equal(param_dict['a'], expected_params[:, :, 0])
    np.testing.assert_array_equal(param_dict['b'], expected_params[:, :, 1])
    np.testing.assert_array_equal(log_likes, expected_log_likes)
    np.testing.assert_array_equal(wts, expected_log_weights)
    assert updater._ess == 0
    assert updater._resampled == True
