import copy
import numpy as np
import pytest

from smcpy.utils.storage import *

class DummyParticles:

    def __init__(self, params, log_likes, log_weights):
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.param_dict = params
        self.params = np.vstack([val for val in params.values()]).T


@pytest.fixture
def mock_particles(mocker):
    mocker.patch('smcpy.utils.storage.Particles', new=DummyParticles)
    mock_particles = mocker.Mock()
    mock_particles.params = np.ones((10, 3))
    mock_particles.param_dict = {'0': np.ones(10), '1': np.ones(10),
                                 '2': np.ones(10)}
    mock_particles.log_likes = np.ones((10, 1))
    mock_particles.log_weights = np.ones((10, 1))
    mock_particles.total_unnorm_log_weight = 99
    mock_particles.attrs = {'phi': 2}
    return mock_particles


def test_inmemorystorage_save(mocker):
    mocks = [mocker.Mock() for _ in range(3)]
    for i, mock in enumerate(mocks):
        mock.attrs = {'phi': i / 10}

    storage = InMemoryStorage()
    storage.save_step(mocks[0])
    storage.save_step(mocks[1])
    storage.save_step(mocks[2])

    assert storage[0] == mocks[0]
    assert storage[1] == mocks[1]
    assert storage[2] == mocks[2]
    assert storage[-1] == mocks[2]
    assert len(storage) == 3
    assert storage.phi_sequence == [0, 0.1, 0.2]

    with pytest.raises(IndexError):
        storage[3]
    with pytest.raises(IndexError):
        storage[-4]


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

def test_hdf5storage_restart(tmpdir, mocker):
    f = tmpdir / 'test.h5'
    f.write_text('test', encoding='ascii')
    mocker.patch('smcpy.utils.storage.HDF5Storage._init_length_on_restart')
    storage = HDF5Storage(str(f))
    assert storage.is_restart

def test_hdf5storage_save(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    def test_particles(p):
        np.testing.assert_array_equal(p.params, mock_particles.params)
        np.testing.assert_array_equal(p.log_likes, mock_particles.log_likes)
        np.testing.assert_array_equal(p.log_weights, mock_particles.log_weights)
        for key, val in mock_particles.param_dict.items():
            np.testing.assert_array_equal(p.param_dict[key], val)
        assert p._total_unlw == 99

    assert isinstance(storage[0], DummyParticles)
    [test_particles(p) for p in storage]
    assert storage.phi_sequence == [2, 2]


def test_hdf5storage_load_existing(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir/'test.h5')
    mp2 = copy.deepcopy(mock_particles)
    mp2.attrs['phi'] = 1
    [storage.save_step(mock_particles) for _ in range(12)] # 2 digits tests sort
    storage.save_step(mp2)

    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles)

    assert storage.phi_sequence == [2] * 12 + [1, 2]


def test_hdf5storage_length(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    assert len(storage) == 2


def test_hdf5storage_neg_indexing(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir/'test.h5')
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    assert storage[-1]
    assert storage[-2]
    assert storage[0]
    assert storage[1]

    with pytest.raises(IndexError):
        storage[2]
    with pytest.raises(IndexError):
        storage[-3]


def test_overwrite_mode(mocker, tmpdir):
    h5_mock = mocker.patch('smcpy.utils.storage.h5py.File')

    filename = tmpdir/'test.h5'
    storage = HDF5Storage(filename=filename, mode='a')

    storage._open_h5('w')

    h5_mock.assert_called_once_with(filename, 'w')
