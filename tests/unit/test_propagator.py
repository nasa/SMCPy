import numpy as np
import pytest

from smcpy.smc.propagator import Propagator

assert_array_equal = np.testing.assert_array_equal


def model(x):
    return np.c_[2 * x[:, 0], x[:, 0]]


@pytest.fixture
def particles(mocker):
    particles = mocker.Mock()
    particles.params = np.ones([5, 1]) * 2
    particles.log_likes = np.ones([5, 1]) * 3
    particles.log_weights = np.ones([5, 1]) * 4
    return particles


@pytest.mark.parametrize('output_names', [None, ('y1', 'y2')])
def test_propagator(particles, output_names, mocker):

    mockParticles = mocker.Mock()
    mocker.patch('smcpy.smc.propagator.Particles', new=mockParticles)

    p = Propagator()
    _ = p.propagate(model, particles, output_names=output_names)

    expected_params = {'y0': particles.params.flatten() * 2,
                       'y1': particles.params.flatten()}

    if output_names is None:
        output_names = ('y0', 'y1')

    assert_array_equal(mockParticles.call_args[0][0][output_names[0]],
                       expected_params['y0'])
    assert_array_equal(mockParticles.call_args[0][0][output_names[1]],
                       expected_params['y1'])
    assert_array_equal(mockParticles.call_args[0][1], particles.log_likes)
    assert_array_equal(mockParticles.call_args[0][2], particles.log_weights)


def test_propagator_bad_output_names(particles):
    p = Propagator()
    with pytest.raises(ValueError):
        _ = p.propagate(model, particles, output_names=['only one'])

