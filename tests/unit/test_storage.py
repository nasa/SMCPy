import copy
import numpy as np
import pytest

from pathlib import Path

from smcpy.utils.storage import *


class DummyParticles:
    def __init__(self, params, log_likes, log_weights):
        self.attrs = {}
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.param_dict = params
        self.params = np.vstack([val for val in params.values()]).T


@pytest.fixture
def mock_particles(mocker):
    mocker.patch("smcpy.utils.storage.Particles", new=DummyParticles)
    mock_particles = mocker.Mock()
    mock_particles.params = np.array([[2] * 10, [0] * 10, [1] * 10]).T
    mock_particles.param_dict = {
        "1": np.ones(10) * 2,
        "0": np.zeros(10),
        "2": np.ones(10),
    }
    mock_particles.log_likes = np.ones((10, 1))
    mock_particles.log_weights = np.ones((10, 1))
    mock_particles.total_unnorm_log_weight = 99
    mock_particles.attrs = {"phi": 2, "mutation_ratio": 1}
    return mock_particles


from unittest.mock import Mock


class DummyClass:
    def __init__(self):
        self.zeros = np.zeros(10)

    def get_ones(self):
        return np.ones(10)


@pytest.fixture
def mock_nested_classes(mocker):
    mocker.patch("smcpy.utils.storage.Particles", new=DummyParticles)

    dummy_class1 = DummyClass()

    param_dict = {
        "1": dummy_class1,
    }
    log_likes = np.ones((10, 1))
    log_weights = np.ones((10, 1))
    mock_particles = DummyParticles(param_dict, log_likes, log_weights)
    mock_particles.total_unnorm_log_weight = 99
    mock_particles.attrs = {"phi": 2, "mutation_ratio": 1}
    return mock_particles


def test_inmemorystorage_save(mocker):
    mocks = [mocker.Mock() for _ in range(3)]
    for i, mock in enumerate(mocks):
        mock.attrs = {"phi": i / 10, "mutation_ratio": i / len(mocks)}

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
    assert storage.mut_ratio_sequence == [0, 1 / 3, 2 / 3]

    with pytest.raises(IndexError):
        storage[3]
    with pytest.raises(IndexError):
        storage[-4]


@pytest.mark.parametrize("steps", [2, 3, 4, 10])
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
    f = tmpdir / "test.txt"
    storage = HDF5Storage(str(f))
    assert not storage.is_restart


def test_hdf5storage_file_not_exist_write(tmpdir):
    f = tmpdir / "test.txt"
    storage = HDF5Storage(str(f), mode="w")
    assert not storage.is_restart


def test_hdf5storage_no_restart_write_mode(tmpdir):
    f = tmpdir / "test.txt"
    f.write_text("test", encoding="ascii")
    storage = HDF5Storage(str(f), mode="w")
    assert not storage.is_restart


def test_hdf5storage_restart(tmpdir, mocker):
    f = tmpdir / "test.h5"
    f.write_text("test", encoding="ascii")
    mocker.patch("smcpy.utils.storage.HDF5Storage._init_length_on_restart")
    storage = HDF5Storage(str(f))
    assert storage.is_restart


def test_hdf5storage_save(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir / "test.h5")
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    def test_particles(p):
        np.testing.assert_array_equal(p.params, mock_particles.params)
        np.testing.assert_array_equal(p.log_likes, mock_particles.log_likes)
        np.testing.assert_array_equal(p.log_weights, mock_particles.log_weights)
        for key, val in mock_particles.param_dict.items():
            np.testing.assert_array_equal(p.param_dict[key], val)
        assert p.attrs == {"phi": 2, "mutation_ratio": 1, "total_unnorm_log_weight": 99}

    assert isinstance(storage[0], DummyParticles)
    [test_particles(p) for p in storage]
    assert storage.phi_sequence == [2, 2]
    assert storage.mut_ratio_sequence == [1, 1]


def test_hdf5storage_load_existing(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir / "test.h5")
    mp2 = copy.deepcopy(mock_particles)
    mp2.attrs["phi"] = 1
    mp2.attrs["mutation_ratio"] = 0
    [storage.save_step(mock_particles) for _ in range(12)]  # 2 digits tests sort
    storage.save_step(mp2)

    storage = HDF5Storage(filename=tmpdir / "test.h5")
    storage.save_step(mock_particles)

    assert storage.phi_sequence == [2] * 12 + [1, 2]
    assert storage.mut_ratio_sequence == [1] * 12 + [0, 1]


def test_hdf5storage_length(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir / "test.h5")
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    assert len(storage) == 2


def test_hdf5storage_neg_indexing(tmpdir, mock_particles):
    storage = HDF5Storage(filename=tmpdir / "test.h5")
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


def test_hdf5storage_overwrite_mode(mocker, tmpdir):
    h5_mock = mocker.patch("smcpy.utils.storage.h5py.File")

    filename = tmpdir / "test.h5"
    storage = HDF5Storage(filename=filename, mode="a")

    storage._open_h5("w")

    h5_mock.assert_called_once_with(filename, "w", track_order=True)


def test_hdf5storage_os_scandir_params(mocker, tmpdir):
    filename = tmpdir / "test.h5"

    mock_scandir = mocker.patch("smcpy.utils.storage.os.scandir")
    storage = HDF5Storage(filename=filename)
    storage._open_h5("a")

    mock_scandir.assert_called_once_with(tmpdir)


@pytest.mark.parametrize("mode", [1, 1000, "b", "abc", "A", "W"])
def test_hdf5storage_invalid_input_mode(tmpdir, mode):
    filename = tmpdir / "test.h5"

    with pytest.raises(ValueError):
        HDF5Storage(filename=filename, mode=mode)


def test_hdf5storage_mode_default(tmpdir):
    filename = tmpdir / "test.h5"
    storage = HDF5Storage(filename=filename)

    assert storage._mode == "a"


@pytest.mark.parametrize("mode", ["w", "a"])
def test_hdf5storage_mode_write(tmpdir, mode):
    filename = tmpdir / "test.h5"
    storage = HDF5Storage(filename=filename, mode=mode)

    assert storage._mode == mode


def test_hdf5storage_first_save_step_changes_mode(tmpdir, mock_particles):
    filename = tmpdir / "test.h5"
    storage = HDF5Storage(filename=filename, mode="w")

    storage.save_step(mock_particles)
    assert storage._mode == "a"


def test_picklestorage_no_restart_file_not_exist(tmpdir):
    f = tmpdir / "test.txt"
    storage = PickleStorage(str(f))
    assert not storage.is_restart


def test_picklestorage_file_not_exist_write(tmpdir):
    f = tmpdir / "test.txt"
    storage = PickleStorage(str(f), mode="wb")
    assert not storage.is_restart


def test_picklestorage_no_restart_write_mode(tmpdir):
    f = tmpdir / "test.txt"
    f.write_text("test", encoding="ascii")
    storage = PickleStorage(str(f), mode="wb")
    assert not storage.is_restart


def test_picklestorage_restart(tmpdir, mocker):
    f = tmpdir / "test.pkl"
    f.write_text("test", encoding="ascii")
    mocker.patch("smcpy.utils.storage.PickleStorage._init_length_on_restart")
    storage = PickleStorage(str(f))
    assert storage.is_restart


def test_picklestorage_save(tmpdir, mock_nested_classes):
    storage = PickleStorage(filename=tmpdir / "test.pkl")
    storage.save_step(mock_nested_classes)
    storage.save_step(mock_nested_classes)

    def test_particles(p):
        np.testing.assert_array_equal(
            p.params[0][0].get_ones(), mock_nested_classes.params[0][0].get_ones()
        )
        np.testing.assert_array_equal(p.log_likes, mock_nested_classes.log_likes)
        np.testing.assert_array_equal(p.log_weights, mock_nested_classes.log_weights)
        for key, dummy_class in mock_nested_classes.param_dict.items():
            get_dummy_class_particles = p.param_dict[key]
            np.testing.assert_array_equal(
                get_dummy_class_particles.get_ones(), dummy_class.get_ones()
            )
            np.testing.assert_array_equal(
                get_dummy_class_particles.zeros, dummy_class.zeros
            )
        assert p.attrs == {"phi": 2, "mutation_ratio": 1, "total_unnorm_log_weight": 99}

    assert isinstance(storage[0], DummyParticles)
    [test_particles(p) for p in storage]
    assert storage.phi_sequence == [2, 2]
    assert storage.mut_ratio_sequence == [1, 1]


def test_picklestorage_load_existing(tmpdir, mock_nested_classes):
    storage = PickleStorage(filename=tmpdir / "test.pkl")
    mp2 = copy.deepcopy(mock_nested_classes)
    mp2.attrs["phi"] = 1
    mp2.attrs["mutation_ratio"] = 0
    [storage.save_step(mock_nested_classes) for _ in range(12)]  # 2 digits tests sort
    storage.save_step(mp2)

    storage = PickleStorage(filename=tmpdir / "test.pkl")
    storage.save_step(mock_nested_classes)

    assert storage.phi_sequence == [2] * 12 + [1, 2]
    assert storage.mut_ratio_sequence == [1] * 12 + [0, 1]


def test_picklestorage_length(tmpdir, mock_nested_classes):
    storage = PickleStorage(filename=tmpdir / "test.pkl")
    storage.save_step(mock_nested_classes)
    storage.save_step(mock_nested_classes)

    assert len(storage) == 2


def test_picklestorage_neg_indexing(tmpdir, mock_nested_classes):
    storage = PickleStorage(filename=tmpdir / "test.pkl")
    storage.save_step(mock_nested_classes)
    storage.save_step(mock_nested_classes)

    assert storage[-1]
    assert storage[-2]
    assert storage[0]
    assert storage[1]

    with pytest.raises(IndexError):
        storage[2]
    with pytest.raises(IndexError):
        storage[-3]


def test_picklestorage_overwrite_mode(mocker, tmpdir):
    pickle_mock = mocker.patch("smcpy.utils.storage.open")

    filename = tmpdir / "test.pkl"
    storage = PickleStorage(filename=filename, mode="ab")

    storage._open_file("wb")

    pickle_mock.assert_called_once_with(filename, "wb")


def test_picklestorage_os_scandir_params(mocker, tmpdir):
    filename = tmpdir / "test.pkl"

    mock_scandir = mocker.patch("smcpy.utils.storage.os.scandir")
    storage = PickleStorage(filename=filename)
    storage._open_file("ab")

    mock_scandir.assert_called_once_with(tmpdir)


@pytest.mark.parametrize("mode", [1, 1000, "b", "abc", "A", "W"])
def test_picklestorage_invalid_input_mode(tmpdir, mode):
    filename = tmpdir / "test.pkl"

    with pytest.raises(ValueError):
        PickleStorage(filename=filename, mode=mode)


def test_picklestorage_mode_default(tmpdir):
    filename = tmpdir / "test.pkl"
    storage = PickleStorage(filename=filename)

    assert storage._mode == "ab"


@pytest.mark.parametrize("mode", ["wb", "ab"])
def test_picklestorage_mode_write(tmpdir, mode):
    filename = tmpdir / "test.pkl"
    storage = PickleStorage(filename=filename, mode=mode)

    assert storage._mode == mode


def test_picklestorage_first_save_step_changes_mode(tmpdir, mock_nested_classes):
    filename = tmpdir / "test.h5"
    storage = PickleStorage(filename=filename, mode="wb")

    storage.save_step(mock_nested_classes)
    assert storage._mode == "ab"
