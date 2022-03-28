import numpy as np
import pytest

from smcpy.utils.storage import *

class DummyParticles:

    def __init__(self, params, log_likes, log_weights):
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.params = params



@pytest.fixture
def mock_particles(mocker):
    mocker.patch('smcpy.utils.storage.Particles', new=DummyParticles)
    mock_particles = mocker.Mock()
    mock_particles.params = np.ones((10, 3))
    mock_particles.log_likes = np.ones((10, 1))
    mock_particles.log_weights = np.ones((10, 1))
    mock_particles.total_unnorm_log_weight = 99
    return mock_particles


def test_inmemorystorage_save():
    storage = InMemoryStorage()
    storage.save_step(1, phi=0)
    storage.save_step(2, phi=0.5)
    storage.save_step(3, phi=1.0)
    assert storage[0] == 1
    assert storage[1] == 2
    assert storage[2] == 3
    assert all([x == i + 1 for i, x in enumerate(storage)])
    assert storage.phi_sequence == [0, 0.5, 1.0]


@pytest.mark.parametrize('steps', [2, 3, 4, 10])
def test_marginal_log_likelihood_calculation(mocker, steps):
    unnorm_log_weights = np.zeros((5, 1))
    expected_mll = [1] + [5 ** (i + 1) for i in range(steps)]

    mocked_particles = mocker.Mock()
    mocked_particles.total_unnorm_log_weight = np.log(5)

    storage = InMemoryStorage()
    storage._step_list = [mocked_particles for _ in range(steps)]

    mll = storage.estimate_marginal_log_likelihoods()
    np.testing.assert_array_almost_equal(mll, np.log(expected_mll))


def test_inmemorystorage_cannot_restart():
    storage = InMemoryStorage()
    assert not storage.is_restart

def test_hdf5storage_no_restart_file_not_exist(tmpdir):
    f = tmpdir / 'test.txt'
    storage = HDF5Storage(str(f))
    assert not storage.is_restart

def test_hdf5storage_no_restart_write_mode(tmpdir):
    f = tmpdir / 'test.txt'
    f.write_text('test', encoding='ascii')
    storage = HDF5Storage(str(f), mode='w')
    assert not storage.is_restart

def test_hdf5storage_restart(tmpdir):
    f = tmpdir / 'test.h5'
    f.write_text('test', encoding='ascii')
    storage = HDF5Storage(str(f))
    assert storage.is_restart

def test_hdf5storage_save(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles, phi=0)
    storage.save_step(mock_particles, phi=1)

    def test_particles(p):
        np.testing.assert_array_equal(p.params, mock_particles.params)
        np.testing.assert_array_equal(p.log_likes, mock_particles.log_likes)
        np.testing.assert_array_equal(p.log_weights, mock_particles.log_weights)
        assert p.total_unnorm_log_weight == 99

    assert isinstance(storage[0], DummyParticles)
    [test_particles(p) for p in storage]
    assert storage.phi_sequence == [0, 1]


def test_hdf5storage_load_existing(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles, phi=0)
    storage.save_step(mock_particles, phi=1)

    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles, phi=0)

    assert storage.phi_sequence == [0, 1, 0]

