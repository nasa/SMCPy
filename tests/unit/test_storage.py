import copy
import numpy as np
import pytest

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
    param_dict = {
        "1": np.ones(10) * 2,
        "0": np.zeros(10),
        "2": np.ones(10),
    }
    log_likes = np.ones((10, 1))
    log_weights = np.ones((10, 1))
    mock_particles = DummyParticles(param_dict, log_likes, log_weights)
    mock_particles.total_unnorm_log_weight = 99
    mock_particles.attrs = {"phi": 2, "mutation_ratio": 1}
    return mock_particles


parametrize_file_storage = pytest.mark.parametrize(
    "storage_type, extension_type", [(HDF5Storage, "h5"), (PickleStorage, "pkl")]
)


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


@parametrize_file_storage
def test_file_storage_no_restart_file_not_exist(tmpdir, storage_type, extension_type):
    f = tmpdir / "test.txt"
    storage = storage_type(str(f))
    assert not storage.is_restart


@parametrize_file_storage
def test_file_storage_not_exist_write(tmpdir, storage_type, extension_type):
    f = tmpdir / "test.txt"
    storage = storage_type(str(f), mode="w")
    assert not storage.is_restart


@parametrize_file_storage
def test_file_storage_no_restart_write_mode(tmpdir, storage_type, extension_type):
    f = tmpdir / "test.txt"
    f.write_text("test", encoding="ascii")
    storage = storage_type(str(f), mode="w")
    assert not storage.is_restart


@parametrize_file_storage
def test_file_storage_restart(tmpdir, mocker, storage_type, extension_type):
    f = tmpdir / f"test.{extension_type}"
    f.write_text("test", encoding="ascii")
    mocker.patch(f"smcpy.utils.storage.{storage_type.__name__}._init_length_on_restart")
    storage = storage_type(str(f))
    assert storage.is_restart


@parametrize_file_storage
def test_file_storage_os_scandir_params(mocker, tmpdir, storage_type, extension_type):
    filename = tmpdir / f"test.{extension_type}"

    mock_scandir = mocker.patch("smcpy.utils.storage.os.scandir")
    storage = storage_type(filename=filename)
    storage._open_file("a")

    mock_scandir.assert_called_once_with(tmpdir)


@parametrize_file_storage
@pytest.mark.parametrize("mode", [1, 1000, "b", "abc", "A", "W"])
def test_file_storage_invalid_input_mode(tmpdir, mode, storage_type, extension_type):
    filename = tmpdir / f"test.{extension_type}"
    with pytest.raises(ValueError):
        storage_type(filename=filename, mode=mode)


@parametrize_file_storage
def test_file_storage_save(tmpdir, mock_particles, storage_type, extension_type):
    storage = storage_type(filename=tmpdir / f"test.{extension_type}")
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


@parametrize_file_storage
def test_file_storage_load_existing(
    tmpdir, mock_particles, storage_type, extension_type
):
    storage = storage_type(filename=tmpdir / f"test.{extension_type}")
    mp2 = copy.deepcopy(mock_particles)
    mp2.attrs["phi"] = 1
    mp2.attrs["mutation_ratio"] = 0
    [storage.save_step(mock_particles) for _ in range(12)]  # 2 digits tests sort
    storage.save_step(mp2)

    storage = storage_type(filename=tmpdir / f"test.{extension_type}")
    storage.save_step(mock_particles)

    assert storage.phi_sequence == [2] * 12 + [1, 2]
    assert storage.mut_ratio_sequence == [1] * 12 + [0, 1]


@parametrize_file_storage
def test_file_storage_length(tmpdir, mock_particles, storage_type, extension_type):
    storage = storage_type(filename=tmpdir / f"test.{extension_type}")
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    assert len(storage) == 2


@parametrize_file_storage
def test_file_storage_neg_indexing(
    tmpdir, mock_particles, storage_type, extension_type
):
    storage = storage_type(filename=tmpdir / f"test.{extension_type}")
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
    storage = HDF5Storage(filename=filename)

    storage._open_file("w")

    h5_mock.assert_called_once_with(filename, "w", track_order=True)


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


def test_picklestorage_overwrite_mode(mocker, tmpdir):
    pickle_mock = mocker.patch("smcpy.utils.storage.open")

    filename = tmpdir / "test.pkl"
    storage = PickleStorage(filename=filename)

    storage._open_file("wb")

    pickle_mock.assert_called_once_with(filename, "wb")


def test_picklestorage_mode_default(tmpdir):
    filename = tmpdir / "test.pkl"
    storage = PickleStorage(filename=filename)

    assert storage._mode == "ab"


@pytest.mark.parametrize("mode", ["w", "a"])
def test_picklestorage_mode_write(tmpdir, mode):
    filename = tmpdir / "test.pkl"
    storage = PickleStorage(filename=filename, mode=mode)

    assert storage._mode == mode + "b"


def test_picklestorage_first_save_step_changes_mode(tmpdir, mock_particles):
    filename = tmpdir / "test.h5"
    storage = PickleStorage(filename=filename, mode="w")

    storage.save_step(mock_particles)
    assert storage._mode == "ab"


def test_picklestorage_array_objects(tmpdir, mock_particles):
    storage = PickleStorage(filename=tmpdir / f"test.pkl")
    mock_particles.params = [list() for _ in range(3)]
    storage.save_step(mock_particles)
    storage.save_step(mock_particles)

    def test_changed_array(p):
        for obj in p.params:
            print(obj)
            assert isinstance(obj, list)

    [test_changed_array(p) for p in storage]
